window.onload = run;
function run() {
  // Add labels
const labels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

// React State implementation in Vanilla JS
const useState = (defaultValue) => {
  let value = defaultValue;
  const getValue = () => value;
  const setValue = (newValue) => (value = newValue);
  return [getValue, setValue];
};

// Declare letiables
const numClass = labels.length;
const [session, setSession] = useState(null);
let mySession;
// Configs
const modelName = "yolov5n-seg.onnx";
const modelInputShape = [1, 3, 640, 640];
const topk = 100;
const iouThreshold = 0.4;
const confThreshold = 0.2;
const classThreshold = 0.2;

// wait until opencv.js initialized
// var cv = require('/path_to_node-opencv-master/lib/opencv');
cv["onRuntimeInitialized"] = async () => {
  // create session
  console.log("Hello_session");
  const [yolov5, nms, mask] = await Promise.all([
    ort.InferenceSession.create("model/yolov5n-seg.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("model/nms-yolov5.onnx", { executionProviders: ["wasm"] }),
    ort.InferenceSession.create("model/mask-yolov5-seg.onnx", { executionProviders: ["wasm"] }),
  ]);
  console.log("Hello_session2")

  //warmup main model
  const tensor = new ort.Tensor(
    "float32",
    new Float32Array(modelInputShape.reduce((a, b) => a * b)),
    modelInputShape
  );
  await yolov5.run({ images: tensor });
  mySession = setSession({ net: yolov5, nms: nms, mask: mask });

};
  
let arrayImgs = document.querySelectorAll('img');
console.log(arrayImgs[0].src)
for(let i=0 ;i<arrayImgs.length;i++){
segmentImage(arrayImgs[i].src)
function segmentImage(imageUrll)
{
  let canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 640;
  canvas.id='canvas';

// Detect Image Function
const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  confThreshold,
  classThreshold,
  inputShape
) => {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new ort.Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new ort.Tensor(
    "float32",
    new Float32Array([topk, iouThreshold, confThreshold])
  ); // nms config tensor
  const { output0, output1 } = await session.net.run({ images: tensor });
  // run session and get output layer
  const { selected_idx } = await session.nms.run({
    detection: output0,
    config: config,
  }); // get selected idx from nms

  const boxes = [];
  const overlay = cv.Mat.zeros(modelHeight, modelWidth, cv.CV_8UC4);

  // looping through output
  for (let idx = 0; idx < output0.dims[1]; idx++) {
    if (!selected_idx.data.includes(idx)) continue; // skip if index isn't selected

    const data = output0.data.slice(
      idx * output0.dims[2],
      (idx + 1) * output0.dims[2]
    ); // get rows
    let box = data.slice(0, 4);
    const confidence = data[4]; // detection confidence
    const scores = data.slice(5, 5 + numClass); // classes probability scores
    let score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    score *= confidence; // multiply score by conf
    const color = colors.get(label); // get color

    // filtering by score thresholds
    if (score >= classThreshold) {
      box = overflowBoxes(
        [
          box[0] - 0.5 * box[2], // before upscale x
          box[1] - 0.5 * box[3], // before upscale y
          box[2], // before upscale w
          box[3], // before upscale h
        ],
        maxSize
      ); // keep boxes in maxSize range

      const [x, y, w, h] = overflowBoxes(
        [
          Math.floor(box[0] * xRatio), // upscale left
          Math.floor(box[1] * yRatio), // upscale top
          Math.floor(box[2] * xRatio), // upscale width
          Math.floor(box[3] * yRatio), // upscale height
        ],
        maxSize
      ); // keep boxes in maxSize range

      boxes.push({
        label: labels[label],
        probability: score,
        color: color,
        bounding: [x, y, w, h], // upscale box
      }); // update boxes to draw later

      const mask = new ort.Tensor(
        "float32",
        new Float32Array([
          ...box, // original scale box
          ...data.slice(5 + numClass), // mask data
        ])
      ); // mask input
      const maskConfig = new ort.Tensor(
        "float32",
        new Float32Array([
          maxSize,
          x, // upscale x
          y, // upscale y
          w, // upscale width
          h, // upscale height
          ...Colors.hexToRgba(color, 120), // color in RGBA
        ])
      ); // mask config
      const { mask_filter } = await session.mask.run({
        detection: mask,
        mask: output1,
        config: maskConfig,
      }); // get mask

      const mask_mat = cv.matFromArray(
        mask_filter.dims[0],
        mask_filter.dims[1],
        cv.CV_8UC4,
        mask_filter.data
      ); // mask result to Mat

      cv.addWeighted(overlay, 1, mask_mat, 1, 0, overlay); // Update mask overlay
      mask_mat.delete(); // delete unused Mat
    }
  }

  const mask_img = new ImageData(
    new Uint8ClampedArray(overlay.data),
    overlay.cols,
    overlay.rows
  ); // create image data from mask overlay
  ctx.putImageData(mask_img, 0, 0); // put ImageData data to canvas
  console.log(canvas.toDataURL("image/png"));
  arrayImgs[i].src=(canvas.toDataURL("image/png"));

  input.delete(); // delete unused Mat
  overlay.delete(); // delete unused Mat
};

/**
 * Get divisible image size by stride
 * @param {Number} stride
 * @param {Number} width
 * @param {Number} height
 * @returns {Number[2]} image size [w, h]
 */
const divStride = (stride, width, height) => {
  if (width % stride !== 0) {
    if (width % stride >= stride / 2)
      width = (Math.floor(width / stride) + 1) * stride;
    else width = Math.floor(width / stride) * stride;
  }
  if (height % stride !== 0) {
    if (height % stride >= stride / 2)
      height = (Math.floor(height / stride) + 1) * stride;
    else height = Math.floor(height / stride) * stride;
  }
  return [width, height];
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @param {Number} stride model stride
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  const mat = cv.imread(source); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  const [w, h] = divStride(stride, matC3.cols, matC3.rows);
  cv.resize(matC3, matC3, new cv.Size(w, h));

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(
    matC3,
    matPad,
    0,
    yPad,
    0,
    xPad,
    cv.BORDER_CONSTANT,
    [0, 0, 0, 255]
  ); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();

  return [input, xRatio, yRatio];
};

/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 */
const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};

class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [
          parseInt(result[1], 16),
          parseInt(result[2], 16),
          parseInt(result[3], 16),
          alpha,
        ]
      : null;
  };
}
const colors = new Colors();
function runInference() {
  fetchBlob(imageUrll);

 
  async function fetchBlob(imageUrll) {
    try {
      const response = await fetch(imageUrll);
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`  );
      }
      const blob = await response.blob();
      const dataUrl = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
      const img = new Image();
      img.src = dataUrl;
      img.onload = function () {
        detectImage(
          img,
          canvas,
          mySession,
          topk,
          iouThreshold,
          confThreshold,
          classThreshold,
          modelInputShape
        );
      };
    } catch (error) {
      console.error("Error fetching image:", error);
    }
  }
}
setTimeout(runInference,6000);
}
}
  
}