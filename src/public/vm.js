// VM and Stride-based Tensor Implementation for Distributed Compute Worker

class Tensor {
  constructor(shape, data = null, strides = null, offset = 0) {
    this.shape = shape;
    this.offset = offset;
    this.size = shape.reduce((a, b) => a * b, 1);
    if (data) {
      this.data = data;
    } else {
      this.data = new Float32Array(this.size);
    }
    if (strides) {
      this.strides = strides;
    } else {
      this.strides = Tensor.computeContiguousStrides(shape);
    }
  }

  static computeContiguousStrides(shape) {
    const strides = new Array(shape.length);
    let s = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = s;
      s *= shape[i];
    }
    return strides;
  }

  get(indices) {
    let index = this.offset;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * this.strides[i];
    }
    return this.data[index];
  }

  set(indices, val) {
    let index = this.offset;
    for (let i = 0; i < indices.length; i++) {
      index += indices[i] * this.strides[i];
    }
    this.data[index] = val;
  }

  isContiguous() {
    let s = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      if (this.strides[i] !== s) return false;
      s *= this.shape[i];
    }
    return true;
  }

  contiguous() {
    if (this.isContiguous()) return this;
    const newData = new Float32Array(this.size);
    const newTensor = new Tensor(this.shape, newData);
    const coords = new Array(this.shape.length).fill(0);
    for (let i = 0; i < this.size; i++) {
      newTensor.data[i] = this.get(coords);
      for (let d = this.shape.length - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < this.shape[d]) {
          break;
        }
        coords[d] = 0;
      }
    }
    return newTensor;
  }

  transpose(dim0 = -2, dim1 = -1) {
    const ndim = this.shape.length;
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    const newShape = [...this.shape];
    const newStrides = [...this.strides];

    const tempShape = newShape[dim0];
    newShape[dim0] = newShape[dim1];
    newShape[dim1] = tempShape;

    const tempStride = newStrides[dim0];
    newStrides[dim0] = newStrides[dim1];
    newStrides[dim1] = tempStride;

    return new Tensor(newShape, this.data, newStrides, this.offset);
  }

  reshape(newShape) {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to shape [${newShape.join(",")}]`);
    }
    const cont = this.contiguous();
    return new Tensor(newShape, cont.data);
  }
}

// === Optional WASM SIMD matmul fast path =================================
// A tiny hand-written WAST kernel (src/public/matmul.wat) compiled to wasm and
// embedded as base64. It vectorises the inner product with f32x4 SIMD — ~5-7x
// faster than the JS triple-loop on large matmuls (the bulk of transformer
// compute). Instantiated synchronously (works in Node and Web Workers); if wasm
// or SIMD is unavailable it silently falls back to the JS path below. Numerically
// equivalent to ~1e-6 (only the float accumulation order differs).
const MATMUL_WASM_B64 = "AGFzbQEAAAABCgFgBn9/f39/fwADAgEABQMBABAHEAIDbWVtAgAGbWF0bXVsAAAKtwIBtAIEBH8CewF9An8gBUF8cSEJQQAhBgJAA0AgBiADTg0BQQAhBwJAA0AgByAJTg0B/QwAAAAAAAAAAAAAAAAAAAAAIQpBACEIAkADQCAIIARODQEgACAGIARsIAhqQQJ0aioCAP0TIQsgASAIIAVsIAdqQQJ0aiENIAogCyAN/QAEAP3mAf3kASEKIAhBAWohCAwACwsgAiAGIAVsIAdqQQJ0aiAK/QsEACAHQQRqIQcMAAsLIAkhBwJAA0AgByAFTg0BQwAAAAAhDEEAIQgCQANAIAggBE4NASAMIAAgBiAEbCAIakECdGoqAgAgASAIIAVsIAdqQQJ0aioCAJSSIQwgCEEBaiEIDAALCyACIAYgBWwgB2pBAnRqIAw4AgAgB0EBaiEHDAALCyAGQQFqIQYMAAsLCw==";

let _matmulWasm = undefined; // undefined = not tried, null = unavailable
function _b64ToBytes(b64) {
  if (typeof Buffer !== 'undefined') return new Uint8Array(Buffer.from(b64, 'base64'));
  const bin = atob(b64); const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}
function getMatmulWasm() {
  if (_matmulWasm !== undefined) return _matmulWasm;
  try {
    const inst = new WebAssembly.Instance(new WebAssembly.Module(_b64ToBytes(MATMUL_WASM_B64)));
    const mem = inst.exports.mem;
    _matmulWasm = {
      fn: inst.exports.matmul,
      f32: () => new Float32Array(mem.buffer),
      ensure: (floats) => {
        const need = floats * 4, have = mem.buffer.byteLength;
        if (need > have) mem.grow(Math.ceil((need - have) / 65536));
      }
    };
  } catch (e) {
    _matmulWasm = null; // no wasm/SIMD → JS fallback
  }
  return _matmulWasm;
}
// Returns true if it computed the matmul via WASM. Only handles contiguous,
// offset-0, 2D or batched-3D matmuls above a size threshold; everything else
// (strided/tiny) falls through to JS.
function tryWasmMatmul(out, a, b) {
  const w = getMatmulWasm();
  if (!w) return false;
  if (a.offset !== 0 || b.offset !== 0 || out.offset !== 0) return false;
  if (!a.isContiguous() || !b.isContiguous() || !out.isContiguous()) return false;
  const an = a.shape.length, bn = b.shape.length;
  let batch, M, K, N;
  if (an === 2 && bn === 2) { batch = 1; M = a.shape[0]; K = a.shape[1]; N = b.shape[1]; if (b.shape[0] !== K) return false; }
  else if (an === 3 && bn === 3) { batch = a.shape[0]; M = a.shape[1]; K = a.shape[2]; N = b.shape[2]; if (b.shape[0] !== batch || b.shape[1] !== K) return false; }
  else return false;
  if (batch * M * N * K < 50000) return false; // not worth the copy overhead
  const aLen = batch * M * K, bLen = batch * K * N, cLen = batch * M * N;
  w.ensure(aLen + bLen + cLen);
  const f = w.f32();
  f.set(a.data.subarray(0, aLen), 0);
  f.set(b.data.subarray(0, bLen), aLen);
  const aPtr = 0, bPtr = aLen * 4, cPtr = (aLen + bLen) * 4;
  for (let bb = 0; bb < batch; bb++) {
    w.fn(aPtr + bb * M * K * 4, bPtr + bb * K * N * 4, cPtr + bb * M * N * 4, M, K, N);
  }
  out.data.set(w.f32().subarray(aLen + bLen, aLen + bLen + cLen), 0);
  return true;
}

// Global math operations implementing strided tensor logic
function matmul(out, a, b) {
  if (tryWasmMatmul(out, a, b)) return;
  if (a.shape.length === 2 && b.shape.length === 2) {
    const M = a.shape[0];
    const K = a.shape[1];
    const N = b.shape[1];
    if (b.shape[0] !== K || out.shape[0] !== M || out.shape[1] !== N) {
      throw new Error(`Shape mismatch for matmul: A is [${a.shape}], B is [${b.shape}], Out is [${out.shape}]`);
    }
    for (let i = 0; i < M; i++) {
      const offsetOut = out.offset + i * out.strides[0];
      const offsetA = a.offset + i * a.strides[0];
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const valA = a.data[offsetA + k * a.strides[1]];
          const valB = b.data[b.offset + k * b.strides[0] + j * b.strides[1]];
          sum += valA * valB;
        }
        out.data[offsetOut + j * out.strides[1]] = sum;
      }
    }
  } else if (a.shape.length === 3 && b.shape.length === 3) {
    const B = a.shape[0];
    const M = a.shape[1];
    const K = a.shape[2];
    const N = b.shape[2];
    if (b.shape[0] !== B || b.shape[1] !== K || out.shape[0] !== B || out.shape[1] !== M || out.shape[2] !== N) {
      throw new Error(`Shape mismatch for 3D matmul: A is [${a.shape}], B is [${b.shape}], Out is [${out.shape}]`);
    }
    for (let bIdx = 0; bIdx < B; bIdx++) {
      const offsetA = a.offset + bIdx * a.strides[0];
      const offsetB = b.offset + bIdx * b.strides[0];
      const offsetOut = out.offset + bIdx * out.strides[0];
      for (let i = 0; i < M; i++) {
        const rowOffsetOut = offsetOut + i * out.strides[1];
        const rowOffsetA = offsetA + i * a.strides[1];
        for (let j = 0; j < N; j++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            const valA = a.data[rowOffsetA + k * a.strides[2]];
            const valB = b.data[offsetB + k * b.strides[1] + j * b.strides[2]];
            sum += valA * valB;
          }
          out.data[rowOffsetOut + j * out.strides[2]] = sum;
        }
      }
    }
  } else {
    throw new Error(`Unsupported shapes for matmul: A is [${a.shape}], B is [${b.shape}]`);
  }
}

function broadcastOp(out, a, b, op) {
  if (a.shape.length === b.shape.length && 
      a.shape.every((v, i) => v === b.shape[i]) && 
      a.shape.every((v, i) => v === out.shape[i]) &&
      a.isContiguous() && b.isContiguous() && out.isContiguous()) {
    const size = out.size;
    const outData = out.data;
    const aData = a.data;
    const bData = b.data;
    const aOffset = a.offset;
    const bOffset = b.offset;
    const outOffset = out.offset;
    if (op === 'add') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] + bData[bOffset + i];
    } else if (op === 'sub') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] - bData[bOffset + i];
    } else if (op === 'mul') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] * bData[bOffset + i];
    } else if (op === 'div') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] / bData[bOffset + i];
    }
    return;
  }

  const ndim = out.shape.length;
  const coords = new Array(ndim).fill(0);
  const size = out.size;

  const shapeA = a.shape;
  const shapeB = b.shape;
  const stridesA = a.strides;
  const stridesB = b.strides;
  const offsetA = a.offset;
  const offsetB = b.offset;

  const aAlignedShape = new Array(ndim).fill(1);
  const aAlignedStrides = new Array(ndim).fill(0);
  const diffA = ndim - shapeA.length;
  for (let i = 0; i < shapeA.length; i++) {
    aAlignedShape[i + diffA] = shapeA[i];
    aAlignedStrides[i + diffA] = stridesA[i];
  }

  const bAlignedShape = new Array(ndim).fill(1);
  const bAlignedStrides = new Array(ndim).fill(0);
  const diffB = ndim - shapeB.length;
  for (let i = 0; i < shapeB.length; i++) {
    bAlignedShape[i + diffB] = shapeB[i];
    bAlignedStrides[i + diffB] = stridesB[i];
  }

  for (let i = 0; i < size; i++) {
    let idxA = offsetA;
    let idxB = offsetB;
    for (let d = 0; d < ndim; d++) {
      if (aAlignedShape[d] > 1) idxA += coords[d] * aAlignedStrides[d];
      if (bAlignedShape[d] > 1) idxB += coords[d] * bAlignedStrides[d];
    }

    const valA = a.data[idxA];
    const valB = b.data[idxB];
    let res = 0;
    if (op === 'add') res = valA + valB;
    else if (op === 'sub') res = valA - valB;
    else if (op === 'mul') res = valA * valB;
    else if (op === 'div') res = valA / valB;

    let idxOut = out.offset;
    for (let d = 0; d < ndim; d++) {
      idxOut += coords[d] * out.strides[d];
    }
    out.data[idxOut] = res;

    for (let d = ndim - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < out.shape[d]) break;
      coords[d] = 0;
    }
  }
}

function scalarOp(out, a, scalarVal, op) {
  if (a.isContiguous() && out.isContiguous() && a.size === out.size) {
    const size = out.size;
    const outData = out.data;
    const aData = a.data;
    const aOffset = a.offset;
    const outOffset = out.offset;
    if (op === 'add') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] + scalarVal;
    } else if (op === 'sub') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] - scalarVal;
    } else if (op === 'mul') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] * scalarVal;
    } else if (op === 'div') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = aData[aOffset + i] / scalarVal;
    }
    return;
  }

  const ndim = out.shape.length;
  const coords = new Array(ndim).fill(0);
  const size = out.size;
  for (let i = 0; i < size; i++) {
    let idxA = a.offset;
    for (let d = 0; d < ndim; d++) {
      idxA += coords[d] * a.strides[d];
    }
    const valA = a.data[idxA];
    let res = 0;
    if (op === 'add') res = valA + scalarVal;
    else if (op === 'sub') res = valA - scalarVal;
    else if (op === 'mul') res = valA * scalarVal;
    else if (op === 'div') res = valA / scalarVal;

    let idxOut = out.offset;
    for (let d = 0; d < ndim; d++) {
      idxOut += coords[d] * out.strides[d];
    }
    out.data[idxOut] = res;

    for (let d = ndim - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < out.shape[d]) break;
      coords[d] = 0;
    }
  }
}

function reduceSum(out, a, axis) {
  out.data.fill(0);
  const ndimA = a.shape.length;
  if (axis < 0) axis += ndimA;

  const coordsA = new Array(ndimA).fill(0);
  const sizeA = a.size;

  for (let i = 0; i < sizeA; i++) {
    let idxA = a.offset;
    for (let d = 0; d < ndimA; d++) {
      idxA += coordsA[d] * a.strides[d];
    }
    const valA = a.data[idxA];

    let idxOut = out.offset;
    if (out.shape.length === ndimA) {
      for (let d = 0; d < ndimA; d++) {
        const coordOut = (d === axis) ? 0 : coordsA[d];
        idxOut += coordOut * out.strides[d];
      }
    } else if (out.shape.length === ndimA - 1) {
      let dOut = 0;
      for (let d = 0; d < ndimA; d++) {
        if (d === axis) continue;
        idxOut += coordsA[d] * out.strides[dOut];
        dOut++;
      }
    } else {
      idxOut = out.offset;
    }

    out.data[idxOut] += valA;

    for (let d = ndimA - 1; d >= 0; d--) {
      coordsA[d]++;
      if (coordsA[d] < a.shape[d]) break;
      coordsA[d] = 0;
    }
  }
}

function reduceMean(out, a, axis) {
  reduceSum(out, a, axis);
  const ndimA = a.shape.length;
  if (axis < 0) axis += ndimA;
  const factor = a.shape[axis];

  const size = out.size;
  if (out.isContiguous()) {
    for (let i = 0; i < size; i++) {
      out.data[out.offset + i] /= factor;
    }
  } else {
    const ndim = out.shape.length;
    const coords = new Array(ndim).fill(0);
    for (let i = 0; i < size; i++) {
      let idxOut = out.offset;
      for (let d = 0; d < ndim; d++) {
        idxOut += coords[d] * out.strides[d];
      }
      out.data[idxOut] /= factor;
      for (let d = ndim - 1; d >= 0; d--) {
        coords[d]++;
        if (coords[d] < out.shape[d]) break;
        coords[d] = 0;
      }
    }
  }
}

function unaryOp(out, a, op) {
  const size = out.size;
  if (a.isContiguous() && out.isContiguous() && a.size === out.size) {
    const outData = out.data;
    const aData = a.data;
    const aOffset = a.offset;
    const outOffset = out.offset;
    if (op === 'relu') {
      for (let i = 0; i < size; i++) outData[outOffset + i] = Math.max(0, aData[aOffset + i]);
    } else if (op === 'gelu') {
      const c = 0.7978845608;
      for (let i = 0; i < size; i++) {
        const x = aData[aOffset + i];
        outData[outOffset + i] = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
      }
    }
    return;
  }

  const ndim = out.shape.length;
  const coords = new Array(ndim).fill(0);
  for (let i = 0; i < size; i++) {
    let idxA = a.offset;
    for (let d = 0; d < ndim; d++) idxA += coords[d] * a.strides[d];
    const x = a.data[idxA];
    let res = 0;
    if (op === 'relu') res = Math.max(0, x);
    else if (op === 'gelu') {
      const c = 0.7978845608;
      res = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }

    let idxOut = out.offset;
    for (let d = 0; d < ndim; d++) idxOut += coords[d] * out.strides[d];
    out.data[idxOut] = res;

    for (let d = ndim - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < out.shape[d]) break;
      coords[d] = 0;
    }
  }
}

function unaryGradOp(out, grad, a, op) {
  const size = out.size;
  if (a.isContiguous() && grad.isContiguous() && out.isContiguous() && a.size === out.size) {
    const outData = out.data;
    const gradData = grad.data;
    const aData = a.data;
    const aOffset = a.offset;
    const gradOffset = grad.offset;
    const outOffset = out.offset;
    if (op === 'relu_grad') {
      for (let i = 0; i < size; i++) {
        outData[outOffset + i] = gradData[gradOffset + i] * (aData[aOffset + i] > 0 ? 1 : 0);
      }
    } else if (op === 'gelu_grad') {
      const c = 0.7978845608;
      for (let i = 0; i < size; i++) {
        const x = aData[aOffset + i];
        const g = c * (x + 0.044715 * x * x * x);
        const tanhG = Math.tanh(g);
        const sech2G = 1 - tanhG * tanhG;
        const gPrime = c * (1 + 0.134145 * x * x);
        const geluPrime = 0.5 * (1 + tanhG) + 0.5 * x * sech2G * gPrime;
        outData[outOffset + i] = gradData[gradOffset + i] * geluPrime;
      }
    }
    return;
  }

  const ndim = out.shape.length;
  const coords = new Array(ndim).fill(0);
  for (let i = 0; i < size; i++) {
    let idxA = a.offset;
    let idxGrad = grad.offset;
    for (let d = 0; d < ndim; d++) {
      idxA += coords[d] * a.strides[d];
      idxGrad += coords[d] * grad.strides[d];
    }
    const x = a.data[idxA];
    const gVal = grad.data[idxGrad];
    let res = 0;
    if (op === 'relu_grad') res = gVal * (x > 0 ? 1 : 0);
    else if (op === 'gelu_grad') {
      const c = 0.7978845608;
      const g = c * (x + 0.044715 * x * x * x);
      const tanhG = Math.tanh(g);
      const sech2G = 1 - tanhG * tanhG;
      const gPrime = c * (1 + 0.134145 * x * x);
      const geluPrime = 0.5 * (1 + tanhG) + 0.5 * x * sech2G * gPrime;
      res = gVal * geluPrime;
    }

    let idxOut = out.offset;
    for (let d = 0; d < ndim; d++) idxOut += coords[d] * out.strides[d];
    out.data[idxOut] = res;

    for (let d = ndim - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < out.shape[d]) break;
      coords[d] = 0;
    }
  }
}

function softmax(out, a) {
  const ndim = a.shape.length;
  const N = a.shape[ndim - 1];
  const outerSize = a.size / N;
  const outerShape = a.shape.slice(0, ndim - 1);
  const outerCoords = new Array(ndim - 1).fill(0);

  for (let i = 0; i < outerSize; i++) {
    let maxVal = -Infinity;
    for (let c = 0; c < N; c++) {
      const coords = [...outerCoords, c];
      let idxA = a.offset;
      for (let d = 0; d < ndim; d++) idxA += coords[d] * a.strides[d];
      const val = a.data[idxA];
      if (val > maxVal) maxVal = val;
    }

    let sumExp = 0;
    const exps = new Float32Array(N);
    for (let c = 0; c < N; c++) {
      const coords = [...outerCoords, c];
      let idxA = a.offset;
      for (let d = 0; d < ndim; d++) idxA += coords[d] * a.strides[d];
      const expVal = Math.exp(a.data[idxA] - maxVal);
      exps[c] = expVal;
      sumExp += expVal;
    }

    for (let c = 0; c < N; c++) {
      const coords = [...outerCoords, c];
      let idxOut = out.offset;
      for (let d = 0; d < ndim; d++) idxOut += coords[d] * out.strides[d];
      out.data[idxOut] = exps[c] / sumExp;
    }

    for (let d = ndim - 2; d >= 0; d--) {
      outerCoords[d]++;
      if (outerCoords[d] < outerShape[d]) break;
      outerCoords[d] = 0;
    }
  }
}

function softmaxGrad(out, grad, softmaxOut) {
  const ndim = softmaxOut.shape.length;
  const N = softmaxOut.shape[ndim - 1];
  const outerSize = softmaxOut.size / N;
  const outerShape = softmaxOut.shape.slice(0, ndim - 1);
  const outerCoords = new Array(ndim - 1).fill(0);

  for (let i = 0; i < outerSize; i++) {
    let sumGradProb = 0;
    for (let c = 0; c < N; c++) {
      const coords = [...outerCoords, c];
      let idxGrad = grad.offset;
      let idxS = softmaxOut.offset;
      for (let d = 0; d < ndim; d++) {
        idxGrad += coords[d] * grad.strides[d];
        idxS += coords[d] * softmaxOut.strides[d];
      }
      sumGradProb += grad.data[idxGrad] * softmaxOut.data[idxS];
    }

    for (let c = 0; c < N; c++) {
      const coords = [...outerCoords, c];
      let idxGrad = grad.offset;
      let idxS = softmaxOut.offset;
      let idxOut = out.offset;
      for (let d = 0; d < ndim; d++) {
        idxGrad += coords[d] * grad.strides[d];
        idxS += coords[d] * softmaxOut.strides[d];
        idxOut += coords[d] * out.strides[d];
      }
      out.data[idxOut] = softmaxOut.data[idxS] * (grad.data[idxGrad] - sumGradProb);
    }

    for (let d = ndim - 2; d >= 0; d--) {
      outerCoords[d]++;
      if (outerCoords[d] < outerShape[d]) break;
      outerCoords[d] = 0;
    }
  }
}

function crossEntropy(lossOut, gradOut, logits, targets) {
  const B = logits.shape[0];
  const C = logits.shape[1];
  
  if (targets.shape[0] !== B) {
    throw new Error(`Targets batch size ${targets.shape[0]} does not match logits batch size ${B}`);
  }

  let totalLoss = 0;
  for (let i = 0; i < B; i++) {
    const targetIdx = Math.round(targets.data[targets.offset + i * targets.strides[0]]);
    if (targetIdx < 0 || targetIdx >= C) {
      throw new Error(`Target class index ${targetIdx} is out of bounds (classes: ${C})`);
    }
    
    let maxLogit = -Infinity;
    for (let c = 0; c < C; c++) {
      const val = logits.data[logits.offset + i * logits.strides[0] + c * logits.strides[1]];
      if (val > maxLogit) maxLogit = val;
    }

    let sumExp = 0;
    const exps = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      const expVal = Math.exp(logits.data[logits.offset + i * logits.strides[0] + c * logits.strides[1]] - maxLogit);
      exps[c] = expVal;
      sumExp += expVal;
    }

    const logSumExp = maxLogit + Math.log(sumExp);
    const targetLogit = logits.data[logits.offset + i * logits.strides[0] + targetIdx * logits.strides[1]];
    
    totalLoss += (logSumExp - targetLogit);

    for (let c = 0; c < C; c++) {
      const prob = exps[c] / sumExp;
      const indicator = (c === targetIdx) ? 1.0 : 0.0;
      gradOut.data[gradOut.offset + i * gradOut.strides[0] + c * gradOut.strides[1]] = (prob - indicator) / B;
    }
  }

  lossOut.data[lossOut.offset] = totalLoss / B;
}

function assign(out, a) {
  if (out.size !== a.size) {
    throw new Error(`Size mismatch for assign: out is ${out.size}, in is ${a.size}`);
  }
  if (out.isContiguous() && a.isContiguous()) {
    out.data.set(a.data.subarray(a.offset, a.offset + a.size), out.offset);
    return;
  }
  const ndim = out.shape.length;
  const coords = new Array(ndim).fill(0);
  const size = out.size;
  for (let i = 0; i < size; i++) {
    let idxA = a.offset;
    for (let d = 0; d < ndim; d++) idxA += coords[d] * a.strides[d];
    let idxOut = out.offset;
    for (let d = 0; d < ndim; d++) idxOut += coords[d] * out.strides[d];
    out.data[idxOut] = a.data[idxA];
    
    for (let d = ndim - 1; d >= 0; d--) {
      coords[d]++;
      if (coords[d] < out.shape[d]) break;
      coords[d] = 0;
    }
  }
}

class TensorVM {
  constructor() {
    this.tensors = {};
  }

  static base64ToFloat32Array(base64) {
    if (typeof Buffer !== 'undefined') {
      const buf = Buffer.from(base64, 'base64');
      return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    } else {
      const binaryString = atob(base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      return new Float32Array(bytes.buffer);
    }
  }

  static float32ArrayToBase64(f32Array) {
    if (typeof Buffer !== 'undefined') {
      const buf = Buffer.from(f32Array.buffer, f32Array.byteOffset, f32Array.byteLength);
      return buf.toString('base64');
    } else {
      const bytes = new Uint8Array(f32Array.buffer, f32Array.byteOffset, f32Array.byteLength);
      let binary = '';
      const len = bytes.byteLength;
      for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      return btoa(binary);
    }
  }

  // === FP16 (half-precision) transport ===================================
  //
  // The model weights and gradients are shipped to/from every worker as base64
  // on each step. Encoding them as 16-bit half-floats instead of 32-bit floats
  // halves the payload size and the server-side string memory. Compute still
  // happens in FP32 inside the VM (we decode FP16 -> FP32 on arrival), so this
  // is a transport-only optimisation with no change to the math, only a small
  // rounding error (~1e-3 relative) on the transmitted values.

  // Convert the raw 32-bit pattern of a float into a 16-bit half-float.
  // (Standard round-to-nearest-even half conversion, à la Three.js DataUtils.)
  static _f32bitsToHalf(x) {
    let bits = (x >> 16) & 0x8000;        // sign
    let m = (x >> 12) & 0x07ff;           // mantissa (with rounding bit)
    const e = (x >> 23) & 0xff;           // exponent
    if (e < 103) return bits;             // too small -> signed zero
    if (e > 142) {                        // too large -> Inf / NaN
      bits |= 0x7c00;
      bits |= ((e === 255) ? 0 : 1) && (x & 0x007fffff);
      return bits;
    }
    if (e < 113) {                        // subnormal half
      m |= 0x0800;
      bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
      return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1); // normal half
    bits += m & 1;                        // round to nearest even
    return bits;
  }

  static halfToFloat32(h) {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  // Single-value helper (convenient for tests).
  static float32ToHalf(val) {
    const fbuf = new Float32Array(1);
    fbuf[0] = val;
    return TensorVM._f32bitsToHalf(new Int32Array(fbuf.buffer)[0]);
  }

  static float32ArrayToFloat16Base64(f32) {
    const u16 = new Uint16Array(f32.length);
    const fbuf = new Float32Array(1);
    const ibuf = new Int32Array(fbuf.buffer);
    for (let i = 0; i < f32.length; i++) {
      fbuf[0] = f32[i];
      u16[i] = TensorVM._f32bitsToHalf(ibuf[0]);
    }
    if (typeof Buffer !== 'undefined') {
      return Buffer.from(u16.buffer, u16.byteOffset, u16.byteLength).toString('base64');
    }
    const bytes = new Uint8Array(u16.buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  static float16Base64ToFloat32Array(base64) {
    let u16;
    if (typeof Buffer !== 'undefined') {
      const buf = Buffer.from(base64, 'base64');
      // Copy into a fresh, 2-byte-aligned ArrayBuffer (Buffer pooling can hand
      // back an odd byteOffset that Uint16Array would reject).
      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      u16 = new Uint16Array(ab);
    } else {
      const binaryString = atob(base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
      u16 = new Uint16Array(bytes.buffer);
    }
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) f32[i] = TensorVM.halfToFloat32(u16[i]);
    return f32;
  }

  // Dispatchers used by the trainer/worker so the precision is a single switch.
  static encodeBase64(f32, dtype) {
    return dtype === 'fp16'
      ? TensorVM.float32ArrayToFloat16Base64(f32)
      : TensorVM.float32ArrayToBase64(f32);
  }

  static decodeBase64(base64, dtype) {
    return dtype === 'fp16'
      ? TensorVM.float16Base64ToFloat32Array(base64)
      : TensorVM.base64ToFloat32Array(base64);
  }

  execute(dslText) {
    const lines = dslText.split("\n");
    for (let lineNo = 0; lineNo < lines.length; lineNo++) {
      let line = lines[lineNo].trim();
      if (line === "" || line.startsWith("#") || line.startsWith("//")) {
        continue;
      }
      
      const spaceIdx = line.indexOf(" ");
      if (spaceIdx === -1) {
        throw new Error(`Line ${lineNo + 1}: invalid instruction "${line}"`);
      }
      const op = line.substring(0, spaceIdx).trim();
      const argsStr = line.substring(spaceIdx + 1).trim();
      const args = argsStr.split(",").map(s => s.trim());

      try {
        this.runInstruction(op, args);
      } catch (err) {
        throw new Error(`Line ${lineNo + 1} ("${line}"): ${err.message}`);
      }
    }
  }

  getTensorOrScalar(arg) {
    if (this.tensors[arg]) {
      return this.tensors[arg];
    }
    const val = parseFloat(arg);
    if (!isNaN(val)) {
      return val;
    }
    throw new Error(`Variable or scalar not found: "${arg}"`);
  }

  runInstruction(op, args) {
    switch (op) {
      case 'matmul': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        const b = this.getTensorOrScalar(args[2]);
        matmul(out, a, b);
        break;
      }
      case 'transpose': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        let dim0 = -2;
        let dim1 = -1;
        if (args.length > 2) dim0 = parseInt(args[2]);
        if (args.length > 3) dim1 = parseInt(args[3]);
        const transposedView = a.transpose(dim0, dim1);
        assign(out, transposedView);
        break;
      }
      case 'add':
      case 'sub':
      case 'mul':
      case 'div': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        const bVal = this.getTensorOrScalar(args[2]);
        if (typeof bVal === 'number') {
          scalarOp(out, a, bVal, op);
        } else {
          broadcastOp(out, a, bVal, op);
        }
        break;
      }
      case 'sum': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        const axis = parseInt(args[2]);
        reduceSum(out, a, axis);
        break;
      }
      case 'mean': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        const axis = parseInt(args[2]);
        reduceMean(out, a, axis);
        break;
      }
      case 'reshape': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        const newShape = args.slice(2).map(s => parseInt(s));
        const reshaped = a.reshape(newShape);
        assign(out, reshaped);
        break;
      }
      case 'relu':
      case 'gelu': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        unaryOp(out, a, op);
        break;
      }
      case 'relu_grad':
      case 'gelu_grad': {
        const out = this.getTensorOrScalar(args[0]);
        const grad = this.getTensorOrScalar(args[1]);
        const a = this.getTensorOrScalar(args[2]);
        unaryGradOp(out, grad, a, op);
        break;
      }
      case 'softmax': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        softmax(out, a);
        break;
      }
      case 'softmax_grad': {
        const out = this.getTensorOrScalar(args[0]);
        const grad = this.getTensorOrScalar(args[1]);
        const softmaxOut = this.getTensorOrScalar(args[2]);
        softmaxGrad(out, grad, softmaxOut);
        break;
      }
      case 'cross_entropy': {
        const loss = this.getTensorOrScalar(args[0]);
        const grad = this.getTensorOrScalar(args[1]);
        const logits = this.getTensorOrScalar(args[2]);
        const targets = this.getTensorOrScalar(args[3]);
        crossEntropy(loss, grad, logits, targets);
        break;
      }
      case 'assign': {
        const out = this.getTensorOrScalar(args[0]);
        const a = this.getTensorOrScalar(args[1]);
        assign(out, a);
        break;
      }
      case 'embedding': {
        const out = this.getTensorOrScalar(args[0]);
        const table = this.getTensorOrScalar(args[1]);
        const ids = this.getTensorOrScalar(args[2]);
        embedding(out, table, ids);
        break;
      }
      case 'embedding_grad': {
        const gTable = this.getTensorOrScalar(args[0]);
        const dy = this.getTensorOrScalar(args[1]);
        const ids = this.getTensorOrScalar(args[2]);
        embeddingGrad(gTable, dy, ids);
        break;
      }
      case 'layernorm': {
        const out = this.getTensorOrScalar(args[0]);
        const x = this.getTensorOrScalar(args[1]);
        const gamma = this.getTensorOrScalar(args[2]);
        const beta = this.getTensorOrScalar(args[3]);
        layernorm(out, x, gamma, beta);
        break;
      }
      case 'layernorm_grad': {
        const dx = this.getTensorOrScalar(args[0]);
        const dgamma = this.getTensorOrScalar(args[1]);
        const dbeta = this.getTensorOrScalar(args[2]);
        const dy = this.getTensorOrScalar(args[3]);
        const x = this.getTensorOrScalar(args[4]);
        const gamma = this.getTensorOrScalar(args[5]);
        layernormGrad(dx, dgamma, dbeta, dy, x, gamma);
        break;
      }
      default:
        throw new Error(`Unknown opcode: "${op}"`);
    }
  }
}

// === Embedding (row gather) and its scatter-add gradient ===================
// table: [V, d], ids: [...] integer indices (stored as floats), out: [...ids, d]
function embedding(out, table, ids) {
  const d = table.shape[table.shape.length - 1];
  const n = ids.size;
  for (let i = 0; i < n; i++) {
    const row = Math.round(ids.data[ids.offset + i * ids.strides[0]]);
    if (row < 0 || row >= table.shape[0]) continue; // bounds safety
    const tBase = table.offset + row * table.strides[0];
    const oBase = out.offset + i * out.strides[0];
    for (let j = 0; j < d; j++) {
      out.data[oBase + j * out.strides[1]] = table.data[tBase + j * table.strides[1]];
    }
  }
}
// gTable: [V, d] (zeroed then scatter-added), dy: [...ids, d], ids: [...]
function embeddingGrad(gTable, dy, ids) {
  gTable.data.fill(0);
  const d = gTable.shape[gTable.shape.length - 1];
  const n = ids.size;
  for (let i = 0; i < n; i++) {
    const row = Math.round(ids.data[ids.offset + i * ids.strides[0]]);
    if (row < 0 || row >= gTable.shape[0]) continue; // bounds safety
    const gBase = gTable.offset + row * gTable.strides[0];
    const yBase = dy.offset + i * dy.strides[0];
    for (let j = 0; j < d; j++) {
      gTable.data[gBase + j * gTable.strides[1]] += dy.data[yBase + j * dy.strides[1]];
    }
  }
}

// === LayerNorm over the last dimension =====================================
const LN_EPS = 1e-5;
function layernorm(out, x, gamma, beta) {
  const d = x.shape[x.shape.length - 1];
  const rows = x.size / d;
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += x.data[x.offset + base + j];
    mean /= d;
    let v = 0;
    for (let j = 0; j < d; j++) { const c = x.data[x.offset + base + j] - mean; v += c * c; }
    const rstd = 1 / Math.sqrt(v / d + LN_EPS);
    for (let j = 0; j < d; j++) {
      const xhat = (x.data[x.offset + base + j] - mean) * rstd;
      out.data[base + j] = xhat * gamma.data[gamma.offset + j] + beta.data[beta.offset + j];
    }
  }
}
// Standard LayerNorm backward producing dx, and reduced dgamma/dbeta over rows.
function layernormGrad(dx, dgamma, dbeta, dy, x, gamma) {
  const d = x.shape[x.shape.length - 1];
  const rows = x.size / d;
  dgamma.data.fill(0);
  dbeta.data.fill(0);
  const xhat = new Float32Array(d);
  const dxhat = new Float32Array(d);
  for (let r = 0; r < rows; r++) {
    const base = r * d;
    let mean = 0;
    for (let j = 0; j < d; j++) mean += x.data[x.offset + base + j];
    mean /= d;
    let v = 0;
    for (let j = 0; j < d; j++) { const c = x.data[x.offset + base + j] - mean; v += c * c; }
    const rstd = 1 / Math.sqrt(v / d + LN_EPS);
    let sumDxhat = 0, sumDxhatXhat = 0;
    for (let j = 0; j < d; j++) {
      const xh = (x.data[x.offset + base + j] - mean) * rstd;
      xhat[j] = xh;
      const dyj = dy.data[dy.offset + base + j];
      dbeta.data[j] += dyj;
      dgamma.data[j] += dyj * xh;
      const dxh = dyj * gamma.data[gamma.offset + j];
      dxhat[j] = dxh;
      sumDxhat += dxh;
      sumDxhatXhat += dxh * xh;
    }
    const mDxhat = sumDxhat / d, mDxhatXhat = sumDxhatXhat / d;
    for (let j = 0; j < d; j++) {
      dx.data[base + j] = rstd * (dxhat[j] - mDxhat - xhat[j] * mDxhatXhat);
    }
  }
}

// UMD-style export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Tensor, TensorVM, matmul, broadcastOp, scalarOp, reduceSum, reduceMean, unaryOp, unaryGradOp, softmax, softmaxGrad, crossEntropy, assign, embedding, embeddingGrad, layernorm, layernormGrad };
} else {
  self.Tensor = Tensor;
  self.TensorVM = TensorVM;
}
