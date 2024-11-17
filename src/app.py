import cv2
from flask import Flask, jsonify, redirect, render_template, request, Response
import whisper
import soundfile as sf
import os
from datetime import datetime

# Load pre-trained Whisper model
model = whisper.load_model("small")
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


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
