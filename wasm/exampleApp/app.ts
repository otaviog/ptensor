import p10 from "../build/ptensor/index.js";

let videoStream: MediaStream | null = null;
let videoElement: HTMLVideoElement;
let captureCanvas: HTMLCanvasElement;
let tensorCanvas: HTMLCanvasElement;
let infoElement: HTMLElement;

async function init() {
  console.log("Initializing PTensor...");
  await p10.init();
  console.log("PTensor initialized!");

  videoElement = document.getElementById("video") as HTMLVideoElement;
  captureCanvas = document.getElementById("captureCanvas") as HTMLCanvasElement;
  tensorCanvas = document.getElementById("tensorCanvas") as HTMLCanvasElement;
  infoElement = document.getElementById("info") as HTMLElement;

  const startCameraBtn = document.getElementById("startCamera") as HTMLButtonElement;
  const captureBtn = document.getElementById("capture") as HTMLButtonElement;

  startCameraBtn.addEventListener("click", startCamera);
  captureBtn.addEventListener("click", captureImage);
}

async function startCamera() {
  try {
    infoElement.innerHTML = "<p>Requesting camera access...</p>";

    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });

    videoElement.srcObject = videoStream;

    videoElement.addEventListener("loadedmetadata", () => {
      const captureBtn = document.getElementById("capture") as HTMLButtonElement;
      captureBtn.disabled = false;
      infoElement.innerHTML = "<p>Camera active. Click 'Capture Image' to take a photo.</p>";
    });
  } catch (error) {
    console.error("Error accessing camera:", error);
    infoElement.innerHTML = `<p style="color: red;">Error accessing camera: ${error}</p>`;
  }
}

function captureImage() {
  const width = videoElement.videoWidth;
  const height = videoElement.videoHeight;

  // Set canvas dimensions
  captureCanvas.width = width;
  captureCanvas.height = height;
  tensorCanvas.width = width;
  tensorCanvas.height = height;

  // Draw video frame to capture canvas
  const ctx = captureCanvas.getContext("2d");
  if (!ctx) {
    console.error("Failed to get 2D context");
    return;
  }

  ctx.drawImage(videoElement, 0, 0, width, height);

  // Get image data
  const imageData = ctx.getImageData(0, 0, width, height);
  const pixels = imageData.data; // RGBA pixel data

  infoElement.innerHTML = `<p>Processing image (${width}x${height})...</p>`;

  try {
    // Convert RGBA to RGB and normalize to 0-255
    const rgbData = new Uint8Array(width * height * 3);
    for (let i = 0; i < pixels.length / 4; i++) {
      rgbData[i * 3 + 0] = pixels[i * 4 + 0]; // R
      rgbData[i * 3 + 1] = pixels[i * 4 + 1]; // G
      rgbData[i * 3 + 2] = pixels[i * 4 + 2]; // B
    }

    // Create tensor from RGB data
    // Shape: [height, width, 3] for HWC format
    const tensor = p10.fromArray(rgbData, [height, width, 3]);

    infoElement.innerHTML = `
      <p><strong>Image captured!</strong></p>
      <p>Tensor shape: ${JSON.stringify(tensor.getShape().toArray())}</p>
      <p>Tensor dtype: ${tensor.getDtype().toString()}</p>
      <p>Tensor size: ${tensor.getSize()}</p>
    `;

    // Draw tensor back to canvas
    drawTensorToCanvas(tensor, width, height);

    // Clean up tensor
    tensor.delete();
  } catch (error) {
    console.error("Error processing image:", error);
    infoElement.innerHTML = `<p style="color: red;">Error: ${error}</p>`;
  }
}

function drawTensorToCanvas(tensor: any, width: number, height: number) {
  const ctx = tensorCanvas.getContext("2d");
  if (!ctx) {
    console.error("Failed to get 2D context for tensor canvas");
    return;
  }

  // Create image data
  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  // Get tensor shape
  const shape = tensor.getShape().toArray();
  console.log("Drawing tensor with shape:", shape);

  // For now, we'll need to read the data back from the tensor
  // Since we don't have a getData() method yet, we'll just display
  // a placeholder showing the tensor was created successfully

  // Fill with a gradient to show it's working
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      pixels[idx + 0] = (x / width) * 255;     // R
      pixels[idx + 1] = (y / height) * 255;    // G
      pixels[idx + 2] = 128;                    // B
      pixels[idx + 3] = 255;                    // A
    }
  }

  ctx.putImageData(imageData, 0, 0);

  // Add text overlay
  ctx.fillStyle = "white";
  ctx.strokeStyle = "black";
  ctx.lineWidth = 2;
  ctx.font = "16px monospace";
  const text = "Tensor created (data reading not yet implemented)";
  ctx.strokeText(text, 10, 30);
  ctx.fillText(text, 10, 30);
}

// Initialize when page loads
init().catch((error) => {
  console.error("Initialization error:", error);
  const info = document.getElementById("info");
  if (info) {
    info.innerHTML = `<p style="color: red;">Initialization error: ${error}</p>`;
  }
});
