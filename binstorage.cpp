#include "pxt.h"
#include "Flash.h"

// TODO move this to separate package

namespace settings {
uintptr_t largeStoreStart();
size_t largeStoreSize();
CODAL_FLASH *largeStoreFlash();
} // namespace settings

namespace binstorage {

#define BUFFER_MAGIC 0xcf429c69

/**
 * Returns the maximum allowed size of binstorage buffer.
 */
//%
uint32_t availableSize() {
    if (settings::largeStoreStart() == 0)
        return 0;
    return settings::largeStoreSize() - 8;
}

/**
 * Return the current binstorage in read-only buffer format (if it was formatted as such).
 */
//%
Buffer asBuffer() {
    uintptr_t beg = settings::largeStoreStart();
    if (!beg)
        return NULL;
    BoxedBuffer *buf = (BoxedBuffer *)beg;
    if (buf->vtable != (uint32_t)&pxt::buffer_vt)
        return NULL;
    return buf;
}

/**
 * Erase storage, and set it up as buffer of given size. Returns null if not available.
 */
//%
Buffer formatAsBuffer(uint32_t size) {
    size_t sz = settings::largeStoreSize();
    if (size > sz - 8)
        return NULL;
    uintptr_t beg = settings::largeStoreStart();
    if (!beg)
        return NULL;
    auto flash = settings::largeStoreFlash();

    uintptr_t p = beg;
    uintptr_t end = beg + sz;
    while (p < end) {
        if (flash->erasePage(p))
            return NULL;
        p += flash->pageSize(p);
    }

    uint32_t header[] = {(uint32_t)&pxt::buffer_vt, size};
    if (flash->writeBytes(beg, header, sizeof(header)))
        return NULL;

    return asBuffer();
}

//%
int _write(int offset, Buffer src) {
    auto buf = asBuffer();
    if (!buf)
        return -1;
    if (offset & 7)
        return -2;
    if (offset < 0 || offset + src->length > buf->length)
        return -3;

    auto flash = settings::largeStoreFlash();
    uint32_t len = (src->length + 7) & ~7;
    if (flash->writeBytes((uintptr_t)buf->data + offset, src->data, len))
        return -4;
    return 0;
}

} // namespace binstorage