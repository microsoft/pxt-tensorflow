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

        ModelSize = 0x184,
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
        autoInvokeSamples = 0
        execTime = 0
        outputs = Buffer.create(0)
        lastError: string

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

        start() {
            super.start()
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
            if (sz > binstore.totalSize() - 8)
                return
            this.eraseModel()
            const flash = binstore.addBuffer(sz)
            const pipe = new InPipe()
            this.sendReport(JDPacket.packed(packet.service_command, "H", [pipe.port]))
            let off = 0
            const headBuffer = Buffer.create(8)
            while (true) {
                const buf = pipe.read()
                if (!buf)
                    control.reset()
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
                    control.reset()
                }
                if (off & 7)
                    throw "invalid model stream size"
            }
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
                default:
                    break;
            }
        }
    }

    //% whenUsed
    export const tfliteHost = new TFLiteHost()
}