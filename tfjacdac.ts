namespace jacdac {
    const arenaSizeSettingsKey = "#jd-tflite-arenaSize"

    export class TFLiteHost extends MLHost {
        constructor(agg: SensorAggregatorHost) {
            super("tflite", ModelRunnerModelFormat.TFLite, agg);
        }

        protected invokeModel() {
            try {
                const res = tf.invokeModelF([this.agg.samplesBuffer])
                this.outputs = packArray(res[0], NumberFormat.Float32LE)
            } catch (e) {
                if (typeof e == "string")
                    this.lastError = e
                control.dmesgValue(e)
            }
        }

        protected eraseModel() {
            tf.freeModel()
            binstore.erase()
            settings.remove(arenaSizeSettingsKey)
        }

        protected loadModelImpl() {
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

        get inputShape(): number[] {
            return tf.inputShape(0)
        }

        get outputShape(): number[] {
            return tf.outputShape(0)
        }
    }

    //% whenUsed
    export const tfliteHost = new TFLiteHost(sensorAggregatorHost)
}