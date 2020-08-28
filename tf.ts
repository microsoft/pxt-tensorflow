namespace tf {
    //% shim=tf::_invokeModel
    declare function _invokeModel(input: number[][], shifts: number[]): number[][];

    //% shim=tf::_loadModel
    declare function _loadModel(model: Buffer, arena_size: number): number;

    //% shim=tf::_shape
    declare function _shape(kind: number, idx: number): number[];

    export function loadModel(model: Buffer, arena_size?: number) {
        if (!arena_size) arena_size = 0
        const res = _loadModel(model, arena_size)
        if (res == -1)
            throw "Wrong model version"
        if (res == -2)
            throw "Can't allocate arena"
        if (res != 0)
            throw `Can't load model: ${res}`
    }

    export function invokeModel(input: number[][], shifts: number[] = null): number[][] {
        let idx = 0
        while (true) {
            const exp = inputElements(idx)
            if (!exp) {
                if (idx != input.length)
                    throw `Wrong number of input arrays: ${input.length} expecting: ${idx}`
                break
            }
            const act = input[idx].length
            if (act != exp)
                throw `Wrong number of elements in array ${idx}: ${act} expecting: ${exp}`
            idx++
        }
        const res = _invokeModel(input, shifts || [])
        if (!res)
            throw "Model invocation error"
        return res
    }

    export function inputShape(index: number) {
        return _shape(0, index)
    }

    export function outputShape(index: number) {
        return _shape(1, index)
    }
}
