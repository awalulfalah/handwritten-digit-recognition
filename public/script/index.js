// Neural Network Functions

const relu = (x) => Math.max(0, x);

const softmax = (x) => {
  const maxVal = Math.max(...x);
  const expValues = x.map(val => Math.exp(val - maxVal));
  const sum = expValues.reduce((acc, val) => acc + val, 0);
  return expValues.map(val => val / sum);
};

const dense = (input, weights, biases, activation = relu) =>
  biases.map((bias, index) =>
    activation(
      input.reduce(
        (sum, inputVal, i) => sum + inputVal * weights[i][index],
        bias
      )
    )
  );

const predict = (input) => {
  let output = dense(input, W1, B1);
  output = dense(output, W2, B2);
  output = dense(output, W3, B3, Math.exp);
  return softmax(output);
};

// Image Processing Functions

const imageDataToGrayscale = (imgData) => {
  const grayscaleImg = [];
  for (let y = 0; y < imgData.height; y++) {
    grayscaleImg[y] = [];
    for (let x = 0; x < imgData.width; x++) {
      const offset = y * 4 * imgData.width + 4 * x;
      const alpha = imgData.data[offset + 3];
      grayscaleImg[y][x] = alpha === 0 ? 1 : imgData.data[offset] / 255;
    }
  }
  return grayscaleImg;
};

const reduceImage = (img) => {
  const reducedSize = 28;
  const blockSize = img.length / reducedSize;
  const reducedImg = Array.from({ length: reducedSize }, () =>
    new Array(reducedSize).fill(0)
  );

  for (let y = 0; y < reducedSize; y++) {
    for (let x = 0; x < reducedSize; x++) {
      let sum = 0;
      for (let v = 0; v < blockSize; v++) {
        for (let h = 0; h < blockSize; h++) {
          sum += img[Math.floor(y * blockSize + v)][Math.floor(x * blockSize + h)];
        }
      }
      reducedImg[y][x] = 1 - sum / (blockSize * blockSize);
    }
  }
  return reducedImg;
};

const getShift = (arr) => {
  const sumCoordinates = arr.reduce(
    (acc, row, x) =>
      row.reduce((rowAcc, cell, y) => {
        if (cell > 0) {
          rowAcc.x += x;
          rowAcc.y += y;
          rowAcc.count++;
        }
        return rowAcc;
      }, acc),
    { x: 0, y: 0, count: 0 }
  );

  return sumCoordinates.count > 0
    ? [
        Math.floor(sumCoordinates.x / sumCoordinates.count) - arr.length / 2,
        Math.floor(sumCoordinates.y / sumCoordinates.count) - arr[0].length / 2,
      ]
    : [0, 0];
};

const centralize = (arr) => {
  const [dx, dy] = getShift(arr);
  return arr.map((row, x) =>
    row.map((_, y) => {
      const newX = x + dx;
      const newY = y + dy;
      return arr[newX] && arr[newX][newY] ? arr[newX][newY] : 0;
    })
  );
};

const flatten = (arr) => arr.flat();

// Canvas Drawing

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1;

let isMouseDown = false;
let lastX = 0;
let lastY = 0;
let predictionTimeout = null;

const setupCanvas = () => {
  ctx.lineWidth = 28;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.font = "28px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = "#212121";
  ctx.strokeStyle = "#212121";
};

const clearCanvas = () => {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col";
    element.children[0].children[0].style.height = "0";
  }
};

const drawLine = (fromX, fromY, toX, toY) => {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
};

const updatePredictions = () => {
  const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  
  const isEmpty = !imgData.data.some((val, i) => i % 4 === 3 && val > 0);
  if (isEmpty) return;
  
  const grayscaleImg = imageDataToGrayscale(imgData);
  const reducedImg = reduceImage(grayscaleImg);
  const centralizedImg = centralize(reducedImg);
  const predictions = predict(flatten(centralizedImg));
  const maxPrediction = Math.max(...predictions);

  for (let i = 0; i < predictions.length; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    element.className =
      predictions[i] === maxPrediction
        ? "prediction-col top-prediction"
        : "prediction-col";
  }
};

const handleMouseDown = (event) => {
  isMouseDown = true;
  lastX = event.offsetX / CANVAS_SCALE;
  lastY = event.offsetY / CANVAS_SCALE;
  
  ctx.beginPath();
  ctx.arc(lastX, lastY, 14, 0, Math.PI * 2);
  ctx.fill();
};

const handleMouseMove = (event) => {
  if (!isMouseDown) return;
  drawLine(
    lastX,
    lastY,
    event.offsetX / CANVAS_SCALE,
    event.offsetY / CANVAS_SCALE
  );
  lastX = event.offsetX / CANVAS_SCALE;
  lastY = event.offsetY / CANVAS_SCALE;
};

const handleMouseUp = () => {
  if (!isMouseDown) return;
  isMouseDown = false;
  
  clearTimeout(predictionTimeout);
  predictionTimeout = setTimeout(() => {
    updatePredictions();
  }, 100);
};

const handleMouseOut = (event) => {
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
};

const handleTouchStart = (event) => {
  event.preventDefault();
  const touch = event.touches[0];
  const rect = canvas.getBoundingClientRect();
  lastX = (touch.clientX - rect.left) / CANVAS_SCALE;
  lastY = (touch.clientY - rect.top) / CANVAS_SCALE;
  isMouseDown = true;
  
  ctx.beginPath();
  ctx.arc(lastX, lastY, 14, 0, Math.PI * 2);
  ctx.fill();
};

const handleTouchMove = (event) => {
  event.preventDefault();
  if (!isMouseDown) return;
  
  const touch = event.touches[0];
  const rect = canvas.getBoundingClientRect();
  const currentX = (touch.clientX - rect.left) / CANVAS_SCALE;
  const currentY = (touch.clientY - rect.top) / CANVAS_SCALE;
  
  drawLine(lastX, lastY, currentX, currentY);
  lastX = currentX;
  lastY = currentY;
};

const handleTouchEnd = (event) => {
  event.preventDefault();
  handleMouseUp();
};

// Initialize

setupCanvas();

canvas.addEventListener("mousedown", handleMouseDown);
canvas.addEventListener("mousemove", handleMouseMove);
document.addEventListener("mouseup", handleMouseUp);
document.addEventListener("mouseout", handleMouseOut);

canvas.addEventListener("touchstart", handleTouchStart);
canvas.addEventListener("touchmove", handleTouchMove);
canvas.addEventListener("touchend", handleTouchEnd);

clearButton.addEventListener("click", clearCanvas);

document.addEventListener("keydown", (event) => {
  if (event.key === "c" || event.key === "C") {
    clearCanvas();
  }
});
