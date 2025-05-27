const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");
const sizes = [16, 32, 48, 128];

function generateIcon(size) {
  canvas.width = size;
  canvas.height = size;

  // Background
  ctx.fillStyle = "#1976d2";
  ctx.fillRect(0, 0, size, size);

  // Dollar sign
  ctx.fillStyle = "#ffffff";
  ctx.font = `bold ${size * 0.6}px Arial`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("$", size / 2, size / 2);

  // Chart line
  ctx.strokeStyle = "#4caf50";
  ctx.lineWidth = size * 0.08;
  ctx.beginPath();
  ctx.moveTo(size * 0.2, size * 0.7);
  ctx.lineTo(size * 0.4, size * 0.4);
  ctx.lineTo(size * 0.6, size * 0.6);
  ctx.lineTo(size * 0.8, size * 0.3);
  ctx.stroke();

  return canvas.toDataURL();
}
