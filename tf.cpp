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
    codal_vdmesg(format, true, args);
    return 0;
}

class WTensorFlow {
  public:
    CodalErrorReporter error_reporter;
    tflite::MicroMutableOpResolver<20> op_resolver;
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

    op_resolver.AddConv2D();
    op_resolver.AddDepthwiseConv2D();
    op_resolver.AddMaxPool2D();
    op_resolver.AddReshape();
    op_resolver.AddFullyConnected();
    op_resolver.AddSoftmax();
    op_resolver.AddAveragePool2D(); // 1932
    op_resolver.AddRelu();          // seems implicit in networks
    op_resolver.AddRelu6();         // 864

    // TEST_OP

#if 0
    // START_OP
    op_resolver.AddAbs();                   // 792
    op_resolver.AddAdd();                   // 7808
    op_resolver.AddArgMax();                // 2028
    op_resolver.AddArgMin();                // 2028
    op_resolver.AddCeil();                  // 1100
    op_resolver.AddConcatenation();         // 4088
    op_resolver.AddCos();                   // 792
    op_resolver.AddDequantize();            // 1976
    op_resolver.AddEqual();                 // 9280
    op_resolver.AddFloor();                 // 500
    op_resolver.AddGreater();               // 8032
    op_resolver.AddGreaterEqual();          // 8032
    op_resolver.AddL2Normalization();       // 3268
    op_resolver.AddLess();                  // 8032
    op_resolver.AddLessEqual();             // 8032
    op_resolver.AddLog();                   // 792
    op_resolver.AddLogicalAnd();            // 1216
    op_resolver.AddLogicalNot();            // 792
    op_resolver.AddLogicalOr();             // 1216
    op_resolver.AddLogistic();              // 2080
    op_resolver.AddMaximum();               // 6452
    op_resolver.AddMean();                  // 2396
    op_resolver.AddMinimum();               // 6452
    op_resolver.AddMul();                   // 5844
    op_resolver.AddNeg();                   // 352
    op_resolver.AddNotEqual();              // 9248
    op_resolver.AddPack();                  // 1380
    op_resolver.AddPad();                   // 6604
    op_resolver.AddPadV2();                 // 6604
    op_resolver.AddPrelu();                 // 3896
    op_resolver.AddQuantize();              // 2092
    op_resolver.AddRelu6();                 // 864
    op_resolver.AddResizeNearestNeighbor(); // 2516
    op_resolver.AddRound();                 // 1036
    op_resolver.AddRsqrt();                 // 824
    op_resolver.AddSin();                   // 792
    op_resolver.AddSplit();                 // 2280
    op_resolver.AddSqrt();                  // 824
    op_resolver.AddSquare();                // 792
    op_resolver.AddStridedSlice();          // 6976
    op_resolver.AddSub();                   // 8416
    op_resolver.AddSvdf();                  // 5788
    op_resolver.AddTanh();                  // 2712
    op_resolver.AddUnpack();                // 1128
                             // END_OP
#endif
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

    uint32_t size = arena_size;
    uint32_t max_size = arena_size;

    if (arena_size == 0) {
        size = 2048;
        max_size = 8 * 1024 * 1024;
    }

    while (size <= max_size) {
        DMESG("allocating %d bytes for arena", size);
        arena = (uint8_t *)malloc(size);
        this->model = modelbuf;
        interpreter =
            new tflite::MicroInterpreter(model, op_resolver, arena, size, &error_reporter);

        if (interpreter->AllocateTensors() == 0) {
            DMESG("allocated; %d used", interpreter->arena_used_bytes());
            return 0;
        }

        freeModel();
        size += size >> 2;
    }

    return -2;
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
        return 0;                                                                                  \
    }

#define GET_TENSOR(tp, expr)                                                                       \
    {                                                                                              \
        auto src = tflite::GetTensorData<tp>(tensor);                                              \
        for (unsigned i = 0; i < sz; ++i) {                                                        \
            expr;                                                                                  \
        }                                                                                          \
        return 0;                                                                                  \
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

int setTensorShift(TfLiteTensor *tensor, RefCollection *data, int shift) {
    unsigned sz = tensorElements(tensor);

    if (data->length() != sz)
        return -1;

    auto src = data->getData();

    // this could be done with bitshift but not sure about numeric stability
    float mul = 1.0f;
    while (shift > 0) {
        shift--;
        mul /= 2.0f;
    }
    while (shift < 0) {
        shift++;
        mul *= 2.0f;
    }

    switch (tensor->type) {
    case kTfLiteFloat32:
        SET_TENSOR(float, dst[i] = mul * toFloat(src[i]));

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
    auto intp = tf->interpreter;
    if (!intp)
        return 0;
    if (idx < 0 || idx >= (int)intp->inputs_size())
        return 0;
    return tensorElements(intp->input(idx));
}

//%
RefCollection *_shape(int kind, int idx) {
    auto tf = getWTensorFlow();
    auto intp = tf->interpreter;
    if (!intp)
        return 0;
    if (kind != 0 && kind != 1)
        return 0;
    int numTensors = kind == 0 ? intp->inputs_size() : intp->outputs_size();
    if (idx < 0 || idx >= numTensors)
        return 0;
    auto tensor = kind == 0 ? intp->input_tensor(idx) : intp->output_tensor(idx);
    auto res = Array_::mk();
    registerGCObj(res);
    for (int i = 0; i < tensor->dims->size; ++i) {
        res->head.push(fromInt(tensor->dims->data[i]));
    }
    unregisterGCObj(res);
    return res;
}

//%
RefCollection *_invokeModel(RefCollection *input, RefCollection *shifts) {
    auto tf = getWTensorFlow();

    auto intp = tf->interpreter;
    if (!intp)
        return NULL;

    auto src = (RefCollection **)input->getData();
    for (unsigned i = 0; i < intp->inputs_size(); ++i) {
        int shift = toInt(shifts->getAt(i));
        if (shift) {
            if (setTensorShift(intp->input(i), src[i], shift) != 0)
                return NULL;
        } else {
            if (setTensor(intp->input(i), src[i]) != 0)
                return NULL;
        }
    }

    int err = tf->invokeModel();
    if (err != 0) {
        DMESG("model error: %d", err);
        return NULL;
    }

    auto res = Array_::mk();
    registerGCObj(res);
    res->setLength(intp->outputs_size());
    auto dst = res->getData();
    for (unsigned i = 0; i < res->length(); ++i) {
        auto tmp = Array_::mk();
        dst[i] = (TValue)tmp;
        getTensor(intp->output(i), tmp);
    }
    unregisterGCObj(res);

    return res;
}

//%
int _loadModel(Buffer model, uint32_t arena_size) {
    pxt::gc(1);
    return getWTensorFlow()->loadModel(model, arena_size);
}

/**
 * Free loaded TF model if any.
 */
//%
void freeModel() {
    getWTensorFlow()->freeModel();
}

/**
 * Get size of used arena space in bytes.
 */
//%
uint32_t arenaBytes() {
    auto intp = getWTensorFlow()->interpreter;
    if (!intp)
        return 0;
    else
        return intp->arena_used_bytes();
}

} // namespace tf