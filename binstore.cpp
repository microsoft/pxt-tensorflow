#include "pxt.h"
#include "Flash.h"

// TODO move this to separate package

namespace settings {
uintptr_t largeStoreStart();
size_t largeStoreSize();
CODAL_FLASH *largeStoreFlash();
} // namespace settings

namespace binstore {

/**
 * Returns the maximum allowed size of binstore buffers.
 */
//%
uint32_t totalSize() {
    return settings::largeStoreStart() ? settings::largeStoreSize() : 0;
}

/**
 * Clear storage.
 */
//%
int erase() {
    size_t sz = settings::largeStoreSize();
    uintptr_t beg = settings::largeStoreStart();
    if (!beg)
        return -1;
    auto flash = settings::largeStoreFlash();

    uintptr_t p = beg;
    uintptr_t end = beg + sz;
    while (p < end) {
        DMESG("erase at %p", p);
        if (flash->erasePage(p))
            return -2;
        p += flash->pageSize(p);
    }

    return 0;
}

//%
RefCollection *buffers() {
    auto res = Array_::mk();
    registerGCObj(res);
    uintptr_t p = settings::largeStoreStart();
    if (p) {
        uintptr_t end = p + settings::largeStoreSize();
        for (;;) {
            BoxedBuffer *buf = (BoxedBuffer *)p;
            if (buf->vtable != (uint32_t)&pxt::buffer_vt)
                break;
            res->head.push((TValue)buf);
            p += (8 + buf->length + 7) & ~7;
            if (p >= end)
                break;
        }
    }
    unregisterGCObj(res);

    return res;
}

/**
 * Add a buffer of given size to binstore.
 */
//%
Buffer addBuffer(uint32_t size) {
    uintptr_t p = settings::largeStoreStart();
    if (!p)
        return NULL;

    BoxedBuffer *buf;
    uintptr_t end = p + settings::largeStoreSize();
    for (;;) {
        buf = (BoxedBuffer *)p;
        if (buf->vtable != (uint32_t)&pxt::buffer_vt)
            break;
        p += (8 + buf->length + 7) & ~7;
        if (p >= end)
            return NULL;
    }

    if (buf->vtable + 1 || buf->length + 1)
        return NULL;

    auto flash = settings::largeStoreFlash();
    uint32_t header[] = {(uint32_t)&pxt::buffer_vt, size};
    if (flash->writeBytes(p, header, sizeof(header)))
        return NULL;

    return buf;
}

PXT_DEF_STRING(sNotAligned, "binstore: not aligned")
PXT_DEF_STRING(sOOR, "binstore: out of range")
PXT_DEF_STRING(sWriteError, "binstore: write failure")
PXT_DEF_STRING(sNotErased, "binstore: not erased")

/**
 * Write bytes in a binstore buffer.
 */
//%
void write(Buffer dst, int dstOffset, Buffer src) {
    if (dstOffset & 7)
        pxt::throwValue((TValue)sNotAligned);
    if (dstOffset < 0 || dstOffset + src->length > dst->length)
        pxt::throwValue((TValue)sOOR);

    auto flash = settings::largeStoreFlash();
    uint32_t len = (src->length + 7) & ~7;
    for (unsigned i = 0; i < len; ++i)
        if (dst->data[dstOffset + i] != 0xff)
            pxt::throwValue((TValue)sNotErased);
    if (flash->writeBytes((uintptr_t)dst->data + dstOffset, src->data, len))
        pxt::throwValue((TValue)sWriteError);
}

} // namespace binstore