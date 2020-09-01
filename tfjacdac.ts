namespace jd_class {
    export const TFLITE = 0x13fe118c
}

namespace jacdac {
    export enum TFLiteSampleType { // uint8_t
        U8 = 0x8,
        I8 = 0x88,
        U16 = 0x10,
        I16 = 0x90,
        U32 = 0x20,
        I32 = 0xa0,
    }

    export enum TFLiteCmd {
        /**
         * Argument: model_size bytes uint32_t. Open pipe for streaming in the model. The size of the model has to be declared upfront.
         * The model is streamed over regular pipe data packets, in the `.tflite` flatbuffer format.
         * When the pipe is closed, the model is written all into flash, and the device running the service may reset.
         */
        SetModel = 0x80,

        /**
         * Argument: outputs pipe (bytes). Open channel that can be used to manually invoke the model. When enough data is sent over the `inputs` pipe, the model is invoked,
         * and results are send over the `outputs` pipe.
         */
        Predict = 0x81,
    }

    export enum TFLiteReg {
        /**
         * Set automatic input collection.
         * These settings are stored in flash.
         */
        Inputs = 0x80,

        /**
         * Read-write uint16_t. When register contains `N > 0`, run the model automatically every time new `N` samples are collected.
         * Model may be run less often if it takes longer to run than `N * sampling_interval`.
         * The `outputs` register will stream its value after each run.
         * This register is not stored in flash.
         */
        AutoInvokeEvery = 0x81,

        /** Read-only bytes. Results of last model invocation as `float32` array. */
        Outputs = 0x101,

        /** Read-only dimension uint16_t. The shape of the input tensor. */
        InputShape = 0x180,

        /** Read-only dimension uint16_t. The shape of the output tensor. */
        OutputShape = 0x181,

        /** Read-only μs uint32_t. The time consumed in last model execution. */
        LastRunTime = 0x182,

        /** Read-only bytes uint32_t. Number of RAM bytes allocated for model execution. */
        AllocatedArenaSize = 0x183,

        /** Read-only bytes uint32_t. The size of `.tflite` model in bytes. */
        ModelSize = 0x184,

        /** Read-only uint32_t. Number of input samples collected so far. */
        NumSamples = 0x185,

        /** Read-only bytes uint8_t. Size of a single sample. */
        SampleSize = 0x186,

        /** Read-write uint8_t. When set to `N`, will stream `N` samples as `current_sample` reading. */
        StreamSamples = 0x82,

        /** Read-only bytes. Last collected sample. */
        CurrentSample = 0x187,

        /** Read-only string (bytes). Textual description of last error when running model (if any). */
        LastError = 0x188,
    }

    function packArray(arr: number[], fmt: NumberFormat) {
        const sz = Buffer.sizeOfNumberFormat(fmt)
        const res = Buffer.create(arr.length * sz)
        for (let i = 0; i < arr.length; ++i)
            res.setNumber(fmt, i * sz, arr[i])
        return res
    }


    const arenaSizeSettingsKey = "#jd-tflite-arenaSize"
    const inputsSettingsKey = "#jd-tflite-inputs"

    function numberFmt(stype: TFLiteSampleType) {
        switch (stype) {
            case TFLiteSampleType.U8: return NumberFormat.UInt8LE
            case TFLiteSampleType.I8: return NumberFormat.Int8LE
            case TFLiteSampleType.U16: return NumberFormat.UInt16LE
            case TFLiteSampleType.I16: return NumberFormat.Int16LE
            case TFLiteSampleType.U32: return NumberFormat.UInt32LE
            case TFLiteSampleType.I32: return NumberFormat.Int32LE
        }
    }

    class Collector extends Client {
        private requiredServiceNum: number
        lastSample: Buffer
        private parent: TFLiteHost
        private numElts: number
        private sampleType: TFLiteSampleType
        private sampleMult: number

        handlePacket(packet: JDPacket) {
            if (packet.service_command == (CMD_GET_REG | REG_READING)) {
                this.parent._newData(packet.timestamp, false)
                const arr = packet.data.toArray(numberFmt(this.sampleType))
                for (let i = 0; i < arr.length; ++i)
                    this.lastSample.setNumber(NumberFormat.Float32LE, i << 2, arr[i] * this.sampleMult)
                this.parent._newData(packet.timestamp, true)
            }
        }

        _attach(dev: Device, serviceNum: number) {
            if (this.requiredServiceNum && serviceNum != this.requiredServiceNum)
                return false
            return super._attach(dev, serviceNum)
        }

        constructor(parent: TFLiteHost, config: Buffer) {
            const [serviceClass, serviceNum, sampleSize, sampleType, sampleShift] = config.unpack("IBBBb", 8)
            const devId = config.getNumber(NumberFormat.Int32LE, 0) == 0 ? null : config.slice(0, 8).toHex()
            super("tfcoll", serviceClass, devId)
            this.requiredServiceNum = serviceNum
            this.sampleType = sampleType

            this.sampleMult = 1
            let sh = sampleShift
            while (sh > 0) {
                this.sampleMult /= 2
                sh--
            }
            while (sh < 0) {
                this.sampleMult *= 2
                sh++
            }

            this.numElts = Math.idiv(sampleSize, Buffer.sizeOfNumberFormat(numberFmt(this.sampleType)))
            this.lastSample = Buffer.create(this.numElts * 4)

            this.parent = parent
        }
    }


    export class TFLiteHost extends Host {
        private autoInvokeSamples = 0
        private execTime = 0
        private outputs = Buffer.create(0)
        private lastError: string
        private collectors: Collector[]
        private lastSample: number
        private samplingInterval: number
        private samplesInWindow: number
        private sampleSize: number
        private streamSamples: number
        private samplesBuffer: Buffer
        private numSamples: number
        private lastRunNumSamples: number

        constructor() {
            super("tflite", jd_class.TFLITE);
        }

        get modelBuffer() {
            const bufs = binstore.buffers()
            if (!bufs || !bufs[0]) return null
            if (bufs[0].getNumber(NumberFormat.Int32LE, 0) == -1)
                return null
            return bufs[0]
        }

        get modelSize() {
            const m = this.modelBuffer
            if (m) return m.length
            else return 0
        }

        get inputSettings() {
            return settings.readBuffer(inputsSettingsKey)
        }

        private pushData() {
            this.samplesBuffer.shift(this.sampleSize)
            let off = this.samplesBuffer.length - this.sampleSize
            for (const coll of this.collectors) {
                this.samplesBuffer.write(off, coll.lastSample)
                off += coll.lastSample.length
            }
            this.numSamples++
            if (this.streamSamples > 0) {
                this.streamSamples--
                this.sendLastSample()
            }
        }

        private runModel() {
            if (this.lastError) return
            const numSamples = this.numSamples
            const t0 = control.micros()
            try {
                const res = tf.invokeModelF([this.samplesBuffer])
                this.outputs = packArray(res[0], NumberFormat.Float32LE)
            } catch (e) {
                if (typeof e == "string")
                    this.lastError = e
                control.dmesgValue(e)
            }
            this.execTime = control.micros() - t0
            this.lastRunNumSamples = numSamples
            this.sendReport(JDPacket.from(CMD_GET_REG | TFLiteReg.Outputs, this.outputs))
        }

        _newData(timestamp: number, isPost: boolean) {
            if (!this.lastSample)
                this.lastSample = timestamp
            const d = timestamp - this.lastSample
            let numSamples = Math.idiv(d + (d >> 1), this.samplingInterval)
            if (!numSamples)
                return
            if (isPost) {
                this.lastSample = timestamp
                this.pushData()
                if (this.autoInvokeSamples && this.lastRunNumSamples >= 0 &&
                    this.numSamples - this.lastRunNumSamples >= this.autoInvokeSamples) {
                    this.lastRunNumSamples = -1
                    control.runInBackground(() => this.runModel())
                }
            } else {
                numSamples--
                if (numSamples > 5) numSamples = 5
                while (numSamples-- > 0)
                    this.pushData()
            }
        }

        start() {
            super.start()
            this.loadModel()
            this.configureInputs()
        }

        private eraseModel() {
            tf.freeModel()
            binstore.erase()
            settings.remove(arenaSizeSettingsKey)
        }

        private loadModel() {
            this.lastError = null
            if (!this.modelBuffer) {
                this.lastError = "no model"
                return
            }
            try {
                const sizeHint = settings.readNumber(arenaSizeSettingsKey)
                tf.loadModel(this.modelBuffer, sizeHint)
                if (sizeHint == undefined)
                    settings.writeNumber(arenaSizeSettingsKey, tf.arenaBytes() + 32)
            } catch (e) {
                if (typeof e == "string")
                    this.lastError = e
                control.dmesgValue(e)
            }
        }

        private configureInputs() {
            const config = this.inputSettings
            if (!config)
                return
            /*
            rw inputs @ 0x80 {
                sampling_interval: u16 ms
                samples_in_window: u16
                reserved: u32
            repeats:
                device_id: u64
                service_class: u32
                service_num: u8
                sample_size: u8 bytes
                sample_type: SampleType
                sample_shift: i8
            }
            */

            [this.samplingInterval, this.samplesInWindow] = config.unpack("HH")
            const entrySize = 16
            let off = 8
            for (const coll of this.collectors || [])
                coll.destroy()
            this.collectors = []
            let frameSz = 0
            while (off < config.length) {
                const coll = new Collector(this, config.slice(off, entrySize))
                coll.setRegInt(REG_STREAMING_INTERVAL, this.samplingInterval)
                coll.setRegInt(REG_IS_STREAMING, 255)
                this.collectors.push(coll)
                frameSz += coll.lastSample.length
                off += entrySize
            }
            this.sampleSize = frameSz
            this.samplesBuffer = Buffer.create(this.samplesInWindow * frameSz)
            this.numSamples = 0
            this.lastRunNumSamples = 0
        }

        private readModel(packet: JDPacket) {
            const sz = packet.intData
            console.log(`model ${sz} bytes (of ${binstore.totalSize()})`)
            if (sz > binstore.totalSize() - 8)
                return
            this.eraseModel()
            const flash = binstore.addBuffer(sz)
            const pipe = new InPipe()
            this.sendReport(JDPacket.packed(packet.service_command, "H", [pipe.port]))
            console.log(`pipe ${pipe.port}`)
            let off = 0
            const headBuffer = Buffer.create(8)
            while (true) {
                const buf = pipe.read()
                if (!buf)
                    return
                if (off == 0) {
                    // don't write the header before we finish
                    headBuffer.write(0, buf)
                    binstore.write(flash, 8, buf.slice(8))
                } else {
                    binstore.write(flash, off, buf)
                }
                off += buf.length
                if (off >= sz) {
                    // now that we're done, write the header
                    binstore.write(flash, 0, headBuffer)
                    // and reset, so we're sure the GC heap is not fragmented when we allocate new arena
                    //control.reset()
                    break
                }
                if (off & 7)
                    throw "invalid model stream size"
            }
            pipe.close()
            this.loadModel()
        }

        private sendLastSample() {
            const buf = this.samplesBuffer.slice(this.samplesBuffer.length - this.sampleSize, this.sampleSize)
            this.sendReport(JDPacket.from(TFLiteReg.CurrentSample | CMD_GET_REG, buf))
        }

        handlePacket(packet: JDPacket) {
            this.handleRegInt(packet, TFLiteReg.AllocatedArenaSize, tf.arenaBytes())
            this.handleRegInt(packet, TFLiteReg.LastRunTime, this.execTime)
            this.handleRegInt(packet, TFLiteReg.ModelSize, this.modelSize)
            this.handleRegInt(packet, TFLiteReg.NumSamples, this.numSamples)
            this.handleRegInt(packet, TFLiteReg.SampleSize, this.sampleSize)
            this.handleRegBuffer(packet, TFLiteReg.Outputs, this.outputs)
            this.streamSamples = this.handleRegInt(packet, TFLiteReg.StreamSamples, this.streamSamples)
            this.autoInvokeSamples = this.handleRegInt(packet, TFLiteReg.AutoInvokeEvery, this.autoInvokeSamples)

            let arr: number[]
            switch (packet.service_command) {
                case TFLiteCmd.SetModel:
                    control.runInBackground(() => this.readModel(packet))
                    break
                case TFLiteReg.Inputs | CMD_GET_REG:
                    this.sendReport(JDPacket.from(packet.service_command, this.inputSettings))
                    break
                case TFLiteReg.Inputs | CMD_SET_REG:
                    if (this.inputSettings && packet.data.equals(this.inputSettings))
                        return // already done
                    settings.writeBuffer(inputsSettingsKey, packet.data)
                    this.configureInputs()
                    break
                case TFLiteReg.OutputShape | CMD_GET_REG:
                    arr = tf.outputShape(0)
                case TFLiteReg.InputShape | CMD_GET_REG:
                    arr = arr || tf.inputShape(0)
                    this.sendReport(JDPacket.from(packet.service_command, packArray(arr, NumberFormat.UInt16LE)))
                    break;
                case TFLiteReg.CurrentSample | CMD_GET_REG:
                    this.sendLastSample()
                    break;
                case TFLiteReg.LastError | CMD_GET_REG:
                    this.sendReport(JDPacket.from(packet.service_command, Buffer.fromUTF8(this.lastError || "")))
                    break
                default:
                    break;
            }
        }
    }

    //% whenUsed
    export const tfliteHost = new TFLiteHost()
}