#include "pxt.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

class CodalErrorReporter : public tflite::ErrorReporter {
  public:
    ~CodalErrorReporter() override {}
    int Report(const char *format, va_list args) override;

  private:
    TF_LITE_REMOVE_VIRTUAL_DELETE
};

int CodalErrorReporter::Report(const char *format, va_list args) {
    codal_vdmesg(format, false, args);
    return 0;
}

class WTensorFlow {
  public:
    CodalErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<65> op_resolver;
    tflite::MicroInterpreter *interpreter;
    uint8_t *arena;
    Buffer model;
    bool hasOutput;
    WTensorFlow();
    int loadModel(Buffer model, uint32_t arena_size);
    int invokeModel();
    void freeModel();
};
SINGLETON(WTensorFlow);

WTensorFlow::WTensorFlow() {
    interpreter = NULL;
    arena = NULL;
    model = NULL;
    hasOutput = false;
    registerGC((TValue *)&model, 1);

    op_resolver.AddAbs();
    op_resolver.AddAdd();
    op_resolver.AddArgMax();
    op_resolver.AddArgMin();
    op_resolver.AddAveragePool2D();
    op_resolver.AddCeil();
    op_resolver.AddConcatenation();
    op_resolver.AddConv2D();
    op_resolver.AddCos();
    op_resolver.AddDepthwiseConv2D();
    op_resolver.AddDequantize();
    op_resolver.AddEqual();
    op_resolver.AddFloor();
    op_resolver.AddFullyConnected();
    op_resolver.AddGreater();
    op_resolver.AddGreaterEqual();
    op_resolver.AddL2Normalization();
    op_resolver.AddLess();
    op_resolver.AddLessEqual();
    op_resolver.AddLog();
    op_resolver.AddLogicalAnd();
    op_resolver.AddLogicalNot();
    op_resolver.AddLogicalOr();
    op_resolver.AddLogistic();
    op_resolver.AddMaximum();
    op_resolver.AddMaxPool2D();
    op_resolver.AddMean();
    op_resolver.AddMinimum();
    op_resolver.AddMul();
    op_resolver.AddNeg();
    op_resolver.AddNotEqual();
    op_resolver.AddPack();
    op_resolver.AddPad();
    op_resolver.AddPadV2();
    op_resolver.AddPrelu();
    op_resolver.AddQuantize();
    op_resolver.AddRelu();
    op_resolver.AddRelu6();
    op_resolver.AddReshape();
    op_resolver.AddResizeNearestNeighbor();
    op_resolver.AddRound();
    op_resolver.AddRsqrt();
    op_resolver.AddSin();
    op_resolver.AddSoftmax();
    op_resolver.AddSplit();
    op_resolver.AddSqrt();
    op_resolver.AddSquare();
    op_resolver.AddStridedSlice();
    op_resolver.AddSub();
    op_resolver.AddSvdf();
    op_resolver.AddTanh();
    op_resolver.AddUnpack();
}

void WTensorFlow::freeModel() {
    if (interpreter)
        delete interpreter;
    interpreter = NULL;
    free(arena);
    arena = NULL;
    model = NULL;
    hasOutput = false;
}

int WTensorFlow::loadModel(Buffer modelbuf, uint32_t arena_size) {
    freeModel();

    auto model = tflite::GetModel(modelbuf->data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        DMESG("Model provided is schema version %d not equal to supported version %d.",
              model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    arena = (uint8_t *)malloc(arena_size);
    this->model = modelbuf;
    interpreter =
        new tflite::MicroInterpreter(model, op_resolver, arena, arena_size, &error_reporter);

    if (interpreter->AllocateTensors() != 0) {
        freeModel();
        return -2;
    }

    return 0;
}

int WTensorFlow::invokeModel() {
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status)
        DMESG("failed invoke: %d", invoke_status);
    else
        hasOutput = true;
    return invoke_status;
}

namespace tf {

static int tfTypeSize(TfLiteType tp) {
    switch (tp) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
        return 4;

    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteBool:
        return 1;

    case kTfLiteInt16:
    case kTfLiteFloat16:
        return 2;

    case kTfLiteInt64:
    case kTfLiteComplex64:
    case kTfLiteFloat64:
        return 8;

    default:
        return -1;
    }
}

static int tensorElements(TfLiteTensor *tensor) {
    int tpsz = tfTypeSize(tensor->type);
    // DMESG("te: %p %s -> %d sz=%d", tensor, TfLiteTypeGetName( tensor->type ), tpsz,
    // tensor->bytes);
    if (tpsz < 0)
        return 0;
    return tensor->bytes / tpsz;
}

#define SET_CLAMPED(a, b)                                                                          \
    int v = toInt(src[i]);                                                                         \
    if (v < a)                                                                                     \
        v = a;                                                                                     \
    if (v > b)                                                                                     \
        v = b;                                                                                     \
    dst[i] = v

#define SET_TENSOR(tp, expr)                                                                       \
    {                                                                                              \
        auto dst = tflite::GetTensorData<tp>(tensor);                                              \
        for (unsigned i = 0; i < sz; ++i) {                                                        \
            expr;                                                                                  \
        }                                                                                          \
        return 0;                                                                                 \
    }

#define GET_TENSOR(tp, expr)                                                                       \
    {                                                                                              \
        auto src = tflite::GetTensorData<tp>(tensor);                                              \
        for (unsigned i = 0; i < sz; ++i) {                                                        \
            expr;                                                                                  \
        }                                                                                          \
        return 0;                                                                                 \
    }

int setTensor(TfLiteTensor *tensor, RefCollection *data) {
    unsigned sz = tensorElements(tensor);

    if (data->length() != sz)
        return -1;

    auto src = data->getData();

    switch (tensor->type) {
    case kTfLiteFloat32:
        SET_TENSOR(float, dst[i] = toFloat(src[i]));

    case kTfLiteInt32:
        SET_TENSOR(int32_t, dst[i] = toInt(src[i]));

    case kTfLiteUInt8:
        SET_TENSOR(uint8_t, SET_CLAMPED(0x00, 0xff));

    case kTfLiteInt8:
        SET_TENSOR(int8_t, SET_CLAMPED(-0x80, 0x7f));

    case kTfLiteInt16:
        SET_TENSOR(int16_t, SET_CLAMPED(-0x8000, 0x7fff));

    case kTfLiteFloat16: // TODO
    default:
        return -2;
    }
}

int getTensor(TfLiteTensor *tensor, RefCollection *data) {
    unsigned sz = tensorElements(tensor);
    data->setLength(sz);

    auto dst = data->getData();

    switch (tensor->type) {
    case kTfLiteFloat32:
        GET_TENSOR(float, dst[i] = fromFloat(src[i]));

    case kTfLiteInt32:
        GET_TENSOR(int32_t, dst[i] = fromInt(src[i]));

    case kTfLiteUInt8:
        GET_TENSOR(uint8_t, dst[i] = fromInt(src[i]));

    case kTfLiteInt8:
        GET_TENSOR(int8_t, dst[i] = fromInt(src[i]));

    case kTfLiteInt16:
        GET_TENSOR(int16_t, dst[i] = fromInt(src[i]));

    case kTfLiteFloat16: // TODO
    default:
        return -2;
    }
}

/**
 * Returns number of elements (numbers) in idx-th input tensor.
 */
//%
int inputElements(int idx) {
    auto tf = getWTensorFlow();
    if (!tf->interpreter)
        return 0;
    if (idx < 0 || idx >= (int)tf->interpreter->inputs_size())
        return 0;
    return tensorElements(tf->interpreter->input(idx));
}

//%
RefCollection *_invokeModel(RefCollection *input) {
    auto tf = getWTensorFlow();

    if (!tf->interpreter)
        return NULL;

    auto src = (RefCollection **)input->getData();
    for (unsigned i = 0; i < tf->interpreter->inputs_size(); ++i)
        if (setTensor(tf->interpreter->input(i), src[i]) != 0)
            return NULL;

    int err = tf->invokeModel();
    if (err != 0) {
        DMESG("model error: %d", err);
        return NULL;
    }

    auto res = Array_::mk();
    registerGCObj(res);
    res->setLength(tf->interpreter->outputs_size());
    auto dst = res->getData();
    for (unsigned i = 0; i < res->length(); ++i) {
        auto tmp = Array_::mk();
        dst[i] = (TValue)tmp;
        getTensor(tf->interpreter->output(i), tmp);
    }
    unregisterGCObj(res);

    return res;
}

//%
int _loadModel(Buffer model, uint32_t arena_size) {
    return getWTensorFlow()->loadModel(model, arena_size);
}

/**
 * Free loaded TF model if any.
 */
//%
void freeModel() {
    getWTensorFlow()->freeModel();
}

} // namespace tf