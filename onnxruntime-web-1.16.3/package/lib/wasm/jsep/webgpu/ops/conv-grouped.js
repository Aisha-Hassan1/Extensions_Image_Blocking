"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.createGroupedConvProgramInfoLoader = void 0;
const util_1 = require("../../util");
const types_1 = require("../types");
const common_1 = require("./common");
const conv_1 = require("./conv");
const fuse_utils_1 = require("./fuse-utils");
const createGroupedConvProgramMetadata = (hasBias, cacheHint) => ({
    name: 'GroupedConv',
    inputTypes: hasBias ? [types_1.GpuDataType.default, types_1.GpuDataType.default, types_1.GpuDataType.default] :
        [types_1.GpuDataType.default, types_1.GpuDataType.default],
    cacheHint
});
const createGroupedConvProgramInfo = (inputs, metadata, attributes, squeezeOutputShapeFunction) => {
    const hasBias = inputs.length > 2;
    const processBias = hasBias ? 'value += b[output_channel];' : '';
    const xShape = inputs[0].dims;
    const wShape = inputs[1].dims;
    const outputChannelsPerGroup = wShape[0] / attributes.group;
    const { activationFunction, applyActivation } = (0, fuse_utils_1.getActicationSnippet)(attributes);
    const isChannelLast = attributes.format === 'NHWC';
    const outputShape = (0, conv_1.calculateOutputShape)(xShape, wShape, attributes.dilations, attributes.pads, attributes.strides, isChannelLast);
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const output = (0, common_1.outputVariable)('output', inputs[0].dataType, outputShape);
    const x = (0, common_1.inputVariable)('x', inputs[0].dataType, xShape);
    const w = (0, common_1.inputVariable)('w', inputs[1].dataType, wShape);
    const inputVars = [x, w];
    if (hasBias) {
        inputVars.push((0, common_1.inputVariable)('b', inputs[2].dataType, inputs[2].dims));
    }
    const getShaderSource = (shaderHelper) => `
  const strides: vec2<u32> = vec2(${attributes.strides[0]}u, ${attributes.strides[1]}u);
  const pads: vec2<u32> = vec2(${attributes.pads[0]}u, ${attributes.pads[1]}u);

  ${shaderHelper.declareVariables(...inputVars, output)}

  ${activationFunction}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let outputIndices = ${output.offsetToIndices('global_idx')};
    let batch: u32 = outputIndices[0];
    let output_channel: u32 = outputIndices[${isChannelLast ? 3 : 1}];
    let xRCCorner: vec2<u32> = vec2<u32>(outputIndices[${isChannelLast ? 1 : 2}], outputIndices[${isChannelLast ? 2 : 3}]) * strides - pads;
    let group_id: u32 = output_channel / ${outputChannelsPerGroup}u;

    var value: ${output.type.value} = ${output.type.value}(0);
    for (var wInChannel: u32 = 0u; wInChannel < ${wShape[1]}u; wInChannel++) {
      let input_channel = group_id * ${wShape[1]}u + wInChannel;
      for (var wHeight: u32 = 0u; wHeight < ${wShape[2]}u; wHeight++) {
        let xHeight = xRCCorner.x + wHeight * ${attributes.dilations[0]}u;

        if (xHeight < 0u || xHeight >= ${xShape[isChannelLast ? 1 : 2]}u) {
          continue;
        }

        for (var wWidth: u32 = 0u; wWidth < ${wShape[3]}u; wWidth++) {
          let xWidth = xRCCorner.y + wWidth * ${attributes.dilations[1]}u;
          if (xWidth < 0u || xWidth >= ${xShape[isChannelLast ? 2 : 3]}u) {
            continue;
          }

          let xVal = ${isChannelLast ? x.get('batch', 'xHeight', 'xWidth', 'input_channel') :
        x.get('batch', 'input_channel', 'xHeight', 'xWidth')};
          let wVal = ${w.get('output_channel', 'wInChannel', 'wHeight', 'wWidth')};
          value += xVal*wVal;
        }
      }
    }
    ${processBias}
    ${applyActivation}
    ${output.setByOffset('global_idx', 'value')}
  }`;
    return {
        ...metadata,
        outputs: [{
                dims: squeezeOutputShapeFunction ? squeezeOutputShapeFunction(outputShape) : outputShape,
                dataType: inputs[0].dataType,
                gpuDataType: types_1.GpuDataType.default
            }],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */) })
    };
};
/**
 * naive grouped conv implementation, supports 1d/2d conv
 * @param squeezeOutputShapeFunction - an optional function to squeeze the output shape, only used in conv1d
 */
const createGroupedConvProgramInfoLoader = (inputs, attributes, squeezeOutputShapeFunction) => {
    const metadata = createGroupedConvProgramMetadata(inputs.length > 2, attributes.cacheKey);
    return {
        ...metadata,
        get: () => createGroupedConvProgramInfo(inputs, metadata, attributes, squeezeOutputShapeFunction)
    };
};
exports.createGroupedConvProgramInfoLoader = createGroupedConvProgramInfoLoader;
//# sourceMappingURL=conv-grouped.js.map