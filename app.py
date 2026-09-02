from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

app = Flask(__name__)

# =========================
# Configuration
# =========================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "handwritten_digit_model.onnx"
)

PORT = int(os.getenv("PORT", "5000"))

# =========================
# Load ONNX Model
# =========================

print("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("✅ ONNX model loaded successfully!")
print("Input:", input_name)
print("Output:", output_name)


# =========================
# Image Preprocessing
# =========================

def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")

    # Convert to numpy
    arr = np.array(image)

    # Detect background
    # White background + black digit
    # → invert to MNIST style
    if arr.mean() > 127:
        arr = 255 - arr

    # Remove very small noise
    arr[arr < 50] = 0

    # Find digit pixels
    coords = np.argwhere(arr > 0)

    if coords.size == 0:
        raise ValueError("No digit detected in image.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop digit
    digit = arr[y_min:y_max + 1, x_min:x_max + 1]

    # Convert to PIL
    digit_image = Image.fromarray(digit)

    # Keep aspect ratio
    max_size = 20

    width, height = digit_image.size

    scale = min(
        max_size / width,
        max_size / height
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    digit_array = np.array(digit_image)

    # Create MNIST-style 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Center digit
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = digit_array

    # Normalize
    canvas = canvas.astype(np.float32) / 255.0

    # Shape: (1, 28, 28, 1)
    canvas = canvas.reshape(
        1, 28, 28, 1
    )

    return canvas


# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400

        # Open image
        image = Image.open(file.stream)

        # Preprocess
        processed = preprocess_image(image)

        # ONNX prediction
        result = session.run(
            [output_name],
            {input_name: processed}
        )

        probabilities = result[0][0]

        # Predicted digit
        predicted_digit = int(
            np.argmax(probabilities)
        )

        # Confidence
        confidence = float(
            probabilities[predicted_digit]
        ) * 100

        return jsonify({

            "success": True,

            "digit": predicted_digit,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities": [
                round(float(x) * 100, 2)
                for x in probabilities
            ]

        })

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================
# Start Server
# =========================

if __name__ == "__main__":

    print("")
    print("=" * 50)
    print("   DigitVision AI")
    print("   Handwritten Digit Recognition")
    print("=" * 50)
    print("")
    print(f"Model: {MODEL_PATH}")
    print(f"Server: http://127.0.0.1:{PORT}")
    print("")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=True
    )