namespace jd_class {
    export const TFLITE = 0x140f9a78
}

namespace jacdac {
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
         * Read-write uint16_t. When register contains `N > 0`, run the model automatically every time new `N` samples are collected.
         * Model may be run less often if it takes longer to run than `N * sampling_interval`.
         * The `outputs` register will stream its value after each run.
         * This register is not stored in flash.
         */
        AutoInvokeEvery = 0x80,

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

        /** Read-only string (bytes). Textual description of last error when running or loading model (if any). */
        LastError = 0x185,
    }

    function packArray(arr: number[], fmt: NumberFormat) {
        const sz = Buffer.sizeOfNumberFormat(fmt)
        const res = Buffer.create(arr.length * sz)
        for (let i = 0; i < arr.length; ++i)
            res.setNumber(fmt, i * sz, arr[i])
        return res
    }


    const arenaSizeSettingsKey = "#jd-tflite-arenaSize"


    export class TFLiteHost extends Host {
        private autoInvokeSamples = 0
        private execTime = 0
        private outputs = Buffer.create(0)
        private lastError: string
        private lastRunNumSamples: number

        constructor(private agg: SensorAggregatorHost) {
            super("tflite", jd_class.TFLITE);
            agg.newDataCallback = () => {
                if (this.autoInvokeSamples && this.lastRunNumSamples >= 0 &&
                    this.numSamples - this.lastRunNumSamples >= this.autoInvokeSamples) {
                    this.lastRunNumSamples = -1
                    control.runInBackground(() => this.runModel())
                }
            }
        }

        get numSamples() {
            return this.agg.numSamples
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

        private runModel() {
            if (this.lastError) return
            const numSamples = this.numSamples
            const t0 = control.micros()
            try {
                const res = tf.invokeModelF([this.agg.samplesBuffer])
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

        start() {
            super.start()
            this.agg.start()
            this.loadModel()
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

        handlePacket(packet: JDPacket) {
            this.handleRegInt(packet, TFLiteReg.AllocatedArenaSize, tf.arenaBytes())
            this.handleRegInt(packet, TFLiteReg.LastRunTime, this.execTime)
            this.handleRegInt(packet, TFLiteReg.ModelSize, this.modelSize)
            this.handleRegBuffer(packet, TFLiteReg.Outputs, this.outputs)
            this.autoInvokeSamples = this.handleRegInt(packet, TFLiteReg.AutoInvokeEvery, this.autoInvokeSamples)

            let arr: number[]
            switch (packet.service_command) {
                case TFLiteCmd.SetModel:
                    control.runInBackground(() => this.readModel(packet))
                    break
                case TFLiteReg.OutputShape | CMD_GET_REG:
                    arr = tf.outputShape(0)
                case TFLiteReg.InputShape | CMD_GET_REG:
                    arr = arr || tf.inputShape(0)
                    this.sendReport(JDPacket.from(packet.service_command, packArray(arr, NumberFormat.UInt16LE)))
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
    export const tfliteHost = new TFLiteHost(sensorAggregatorHost)
}