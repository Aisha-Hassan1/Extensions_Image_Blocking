"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseSoftmaxAttributes = exports.softmax = exports.softmaxProgramMetadata = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length !== 1) {
        throw new Error('Softmax op requires 1 input.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */) {
        throw new Error('Softmax input needs to be float.');
    }
};
exports.softmaxProgramMetadata = {
    name: 'Softmax',
    inputTypes: [types_1.GpuDataType.default]
};
const createSoftmaxProgramInfo = (input, attributes) => {
    const dataType = 'f32';
    const shape = input.dims;
    const outputSize = util_1.ShapeUtil.size(shape);
    const WG = 64;
    let axis = attributes.axis;
    if (axis < 0) {
        axis = shape.length + axis;
    }
    if (axis < shape.length - 1) {
        throw new Error('softmax only supports last axis for now.');
    }
    const cols = shape[axis];
    const rows = outputSize / cols;
    const getShaderSource = (_shaderHelper) => `
      var<workgroup> rowMaxShared : ${dataType};
      var<workgroup> rowSumShared : ${dataType};
      var<workgroup> threadShared : array<${dataType}, ${WG}>;

      @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
      @group(0) @binding(1) var<storage, read_write> result : array<${dataType}>;

      fn getValue(row: i32, col: i32, row_stride: i32) -> ${dataType} {
        let index = row * row_stride + col;
        return x[index];
      }

      fn setValue(row: i32, col: i32, row_stride: i32, value: ${dataType}) {
        let index = row * row_stride + col;
        result[index] = value;
      }

      @compute @workgroup_size(${WG}, 1, 1)
      fn main(@builtin(local_invocation_id) local_id : vec3<u32>, @builtin(global_invocation_id) global_id : vec3u) {
        let gindex = i32(global_id.x);
        let lindex = i32(local_id.x);
        const wg = ${WG};
        let row = gindex / wg;
        let cols = ${cols};
        let row_stride : i32 = ${cols};

        // find the rows max
        var threadMax = -3.402823e+38f; // 6.2.4 in wgsl spec
        for (var col = lindex; col < cols; col += wg) {
          let value = getValue(row, col, row_stride);
          threadMax = max(threadMax, value);
        }
        if (lindex < cols) {
          threadShared[lindex] = threadMax;
        }
        workgroupBarrier();

        var reduceSize = min(cols, wg);
        for (var currSize = reduceSize >> 1;  currSize > 0; currSize = reduceSize >> 1) {
          reduceSize = currSize + (reduceSize & 1);
          if (lindex < currSize) {
            threadShared[lindex] = max(threadShared[lindex], threadShared[lindex + reduceSize]);
          }
          workgroupBarrier();
        }
        if (lindex == 0) {
          rowMaxShared = threadShared[0];
        }
        workgroupBarrier();

        // find the rows sum
        var threadSum = 0.0;
        for (var col = lindex; col < cols; col += wg) {
          let subExp = exp(getValue(row, col, row_stride) - rowMaxShared);
          threadSum += subExp;
        }
        threadShared[lindex] = threadSum;
        workgroupBarrier();

        for (var currSize = wg >> 1;  currSize > 0; currSize = currSize >> 1) {
          if (lindex < currSize) {
            threadShared[lindex] = threadShared[lindex] + threadShared[lindex + currSize];
          }
          workgroupBarrier();
        }
        if (lindex == 0) {
          rowSumShared = threadShared[0];
        }
        workgroupBarrier();

        // calculate final value for each element in the row
        for (var col = lindex; col < cols; col += wg) {
          let value = exp(getValue(row, col, row_stride) - rowMaxShared) / rowSumShared;
          setValue(row, col, row_stride, value);
        }
      }`;
    return {
        ...exports.softmaxProgramMetadata,
        outputs: [{ dims: shape, dataType: input.dataType, gpuDataType: types_1.GpuDataType.default }],
        getShaderSource,
        dispatchGroup: () => ({ x: rows })
    };
};
const softmax = (context, attributes) => {
    validateInputs(context.inputs);
    context.compute({
        ...exports.softmaxProgramMetadata,
        cacheHint: attributes.cacheKey,
        get: () => createSoftmaxProgramInfo(context.inputs[0], attributes)
    });
};
exports.softmax = softmax;
const parseSoftmaxAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axis: attributes.axis });
exports.parseSoftmaxAttributes = parseSoftmaxAttributes;
//# sourceMappingURL=softmax.js.map