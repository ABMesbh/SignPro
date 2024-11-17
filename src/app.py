from flask import Flask, jsonify, redirect, render_template, request
import whisper
import soundfile as sf
import os
from datetime import datetime
import cv2
from transformers import pipeline
from PIL import Image

# Load pre-trained Whisper model
model = whisper.load_model("small")

# Initialize the OCR pipeline
ocr_pipeline = pipeline("image-to-text", model="kha-white/manga-ocr-base")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            print("No file selected")
            return redirect(request.url)

        if file:
            # Save the file to the server
            
            # Get the absolute path of the project's root directory
            root_path = os.path.dirname(os.path.abspath(__file__))
            # Join the root path with the 'uploads' folder and the file's name
            filepath = os.path.join(root_path, "uploads", file.filename)
            print(file.filename)
            print(filepath)
            file.save(filepath)
            print("Saving file to server")
            data = model.transcribe(filepath, language="fr")    
            transcript = data["text"]
            print(transcript)

            # Return transcription as a JSON response
            return jsonify({"transcription": transcript.strip()[0]})
        

    return render_template('index.html', transcript=transcript)


def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None

    print("Press any key to capture the image...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        cv2.imshow("Press any key to capture", frame)
        if cv2.waitKey(1) & 0xFF != 255:
            cv2.imwrite("uploads/captured_image.png", frame)
            print("Image captured and saved as 'captured_image.png'")
            break

    cap.release()
    cv2.destroyAllWindows()
    # Convert the captured frame to a PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

# Function to detect text using the OCR pipeline
def detect_text(image):
    result = ocr_pipeline(image)
    if result:
        print("Detected text:", result[0]['generated_text'])
        return result[0]['generated_text'][0]
    return ""

# Route to capture an image and detect a letter
@app.route('/capture', methods=['POST'])
def capture_and_detect():
    # Capture the image
    image = capture_image()
    if image is None:
        return jsonify({"error": "Failed to capture image"}), 500

    # Detect the letter using the OCR pipeline
    letter = detect_text(image)
    return jsonify({"letter": letter})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
