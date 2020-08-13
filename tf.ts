namespace tf {
    //% shim=tf::_invokeModel
    declare function _invokeModel(input: number[][]): number[][];

    //% shim=tf::_loadModel
    declare function _loadModel(model: Buffer, arena_size: number): number;

    export function loadModel(model: Buffer, arena_size: number) {
        const res = _loadModel(model, arena_size)
        if (res != 0)
            throw `Can't load model: ${res}`
    }

    export function invokeModel(input: number[][]): number[][] {
        let idx = 0
        while (true) {
            const exp = inputElements(idx)
            if (!exp) {
                if (idx + 1 != input.length)
                    throw `Wrong number of input arrays: ${input.length} expecting: ${idx + 1}`
                break
            }
            const act = input[idx].length
            if (act != exp)
                throw `Wrong number of elements in array ${idx}: ${act} expecting: ${exp}`
        }
        const res = _invokeModel(input)
        if (!res)
            throw "Model invocation error"
        return res
    }
}
