import cv2
from flask import Flask, jsonify, redirect, render_template, request, Response
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
cap = cv2.VideoCapture(0)
app = Flask(__name__)
objectif = ""

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

@app.route("/start", methods=["POST"])
def start():
    global cap, objectif
    objectif = request.form.get("transcript")
    objectif = objectif.lower().strip()[0]
    print("Objectif: ", objectif)
    print("Starting recording")
    return ('', 204)

@app.route("/video_feed")
def rec():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_frames():
    global cap,objectif
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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
    global cap
    # Capture the image
    success, image = cap.read()
    if not success:
        return jsonify({"error": "Failed to capture image"}), 500
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detect the letter using the OCR pipeline
    letter = detect_text(Image.fromarray(img))
    return jsonify({"letter": letter})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
