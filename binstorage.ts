namespace binstorage {
    //% shim=binstorage::_write
    declare function _write(offset: number, src: Buffer): number;

    export function write(offset: number, src: Buffer) {
        const r = _write(offset, src)
        switch (r) {
            case -1:
                throw "not formatted"
            case -2:
                throw "write offset not 8-aligned"
            case -3:
                throw "write offset out of range"
            case -4:
                throw "write error"
        }
    }

} // namespace binstorage