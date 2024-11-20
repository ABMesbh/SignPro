import time
import cv2
from flask import Flask, jsonify, redirect, render_template, request, Response
import whisper
import soundfile as sf
import os
from datetime import datetime
import cv2
from transformers import pipeline
from PIL import Image
import torch
import torch.nn as nn
import mediapipe as mp
import numpy as np
from dataUtils import normalizeData

# Define the hand model
class HandModel(nn.Module):
    def __init__(self):
        super(HandModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=0)
        return x

handmodel = HandModel()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
root_path = os.path.dirname(os.path.abspath(__file__))
# Join the root path with the 'uploads' folder and the file's name
handmodel.load_state_dict(torch.load(os.path.join(root_path, "uploads", "handmodel.pth")))

# laod classes
classes = []
with open(os.path.join(root_path, "uploads", "classes.txt"), "r") as f:
    for line in f:
        classes.append(line.strip())
print(classes)

# Load pre-trained Whisper model
model = whisper.load_model("small")

# Initialize the OCR pipeline
ocr_pipeline = pipeline("image-to-text", model="kha-white/manga-ocr-base")
app = Flask(__name__)

objectif = ""
result = False
cap = None 

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
    global objectif, result, cap
    result = False
    objectif = request.form.get("transcript", "").lower().strip()[0]
    print("Starting learning mode with objectif:", objectif)

    # Open the camera when `/start` is accessed
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        print("Camera opened")
        if not cap.isOpened():
            print("Failed to access camera")
            return jsonify({"error": "Failed to access camera"}), 500

    time.sleep(5)
    return jsonify({"end": "None"})


@app.route("/video_feed")
def video_feed():
    return Response(capture_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def capture_frames():
    global objectif, result, cap
    score = 0
    
    while True:
        i=0
        while cap is None:
            time.sleep(1)
        while cap and cap.isOpened():  
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            if objectif != "" and not result:
                with mp_hands.Hands(
                    model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
                ) as hands:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style(),
                            )

                        a = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                        a = normalizeData(a)
                        a = torch.tensor(a).float()
                        image = cv2.flip(image, 1)
                        output = handmodel(a)
                        print("Output: ", output)
                        objid = classes.index(objectif)
                        id = torch.argmax(output).item()
                        res = torch.max(output).item()
                        if id == objid and res >= 0.98 and objid != 4:
                            print("Objectif atteint")
                            objectif = ""
                            result = True
                            score = res

            elif result and objectif == "":
                # Display the score
                image = cv2.rectangle(image, (0, 0), (int(cap.get(3)), int(cap.get(4) * 0.1)), (0, 0, 0), -1)
                image = cv2.putText(
                    image,
                    "Score: " + str(round(score, 2)),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                i += 1
                if i == 3:
                    cap.release()
                    cap = None
                    break
                    # yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n\r\n")
            ret, buffer = cv2.imencode(".jpg", image)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            
# Function to detect text using the OCR pipeline
def detect_text(image):
    result = ocr_pipeline(image)
    if result:
        print("Detected text:", result[0]['generated_text'])
        return result[0]['generated_text'][0]
    return ""

@app.route('/capture', methods=['POST'])
def capture_and_detect():
    # Ensure an image file is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    
    image_file = request.files['image']

    try:
        # Convert the uploaded image to a format suitable for OpenCV
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to process uploaded image"}), 500

        # Convert the image to RGB format for PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform OCR to detect the letter
        letter = detect_text(Image.fromarray(img_rgb))
        return jsonify({"letter": letter})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
