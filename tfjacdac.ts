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
    }

    function packArray(arr: number[], fmt: NumberFormat) {
        const sz = Buffer.sizeOfNumberFormat(fmt)
        const res = Buffer.create(arr.length * sz)
        for (let i = 0; i < arr.length; ++i)
            res.setNumber(fmt, i * sz, arr[i])
        return res
    }


    export class TFLiteHost extends Host {
        autoInvokeSamples = 0
        execTime = 0
        outputs = Buffer.create(0)

        constructor() {
            super("tflite", jd_class.TFLITE);
        }

        handlePacket(packet: JDPacket) {
            this.handleRegInt(packet, TFLiteReg.AllocatedArenaSize, tf.arenaBytes())
            this.handleRegInt(packet, TFLiteReg.LastRunTime, this.execTime)
            this.handleRegBuffer(packet, TFLiteReg.Outputs, this.outputs)

            this.autoInvokeSamples = this.handleRegInt(packet, TFLiteReg.AutoInvokeEvery, this.autoInvokeSamples)

            let arr: number[]
            switch (packet.service_command) {
                case TFLiteCmd.SetModel:
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