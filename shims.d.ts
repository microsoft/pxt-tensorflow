// Auto-generated. Do not edit.
declare namespace tf {

    /**
     * Returns number of elements (numbers) in idx-th input tensor.
     */
    //% shim=tf::inputElements
    function inputElements(idx: int32): int32;

    /**
     * Free loaded TF model if any.
     */
    //% shim=tf::freeModel
    function freeModel(): void;

    /**
     * Get size of used arena space in bytes.
     */
    //% shim=tf::arenaBytes
    function arenaBytes(): uint32;
}
declare namespace binstore {

    /**
     * Returns the maximum allowed size of binstore buffers.
     */
    //% shim=binstore::totalSize
    function totalSize(): uint32;

    /**
     * Clear storage.
     */
    //% shim=binstore::erase
    function erase(): int32;

    /**
     * Add a buffer of given size to binstore.
     */
    //% shim=binstore::addBuffer
    function addBuffer(size: uint32): Buffer;

    /**
     * Write bytes in a binstore buffer.
     */
    //% shim=binstore::write
    function write(dst: Buffer, dstOffset: int32, src: Buffer): int32;
}

// Auto-generated. Do not edit. Really.
