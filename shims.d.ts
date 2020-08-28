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
declare namespace binstorage {

    /**
     * Returns the maximum allowed size of binstorage buffer.
     */
    //% shim=binstorage::availableSize
    function availableSize(): uint32;

    /**
     * Return the current binstorage in read-only buffer format (if it was formatted as such).
     */
    //% shim=binstorage::asBuffer
    function asBuffer(): Buffer;

    /**
     * Erase storage, and set it up as buffer of given size. Returns null if not available.
     */
    //% shim=binstorage::formatAsBuffer
    function formatAsBuffer(size: uint32): Buffer;
}

// Auto-generated. Do not edit. Really.
