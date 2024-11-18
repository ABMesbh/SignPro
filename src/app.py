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
cap = cv2.VideoCapture(0)
app = Flask(__name__)
objectif = ""
result = False

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
        
    global cap
    cap = cv2.VideoCapture(0)
    return render_template('index.html', transcript=transcript)

@app.route("/start", methods=["POST"])
def start():
    global cap, objectif,result
    result = False
    objectif = request.form.get("transcript")
    objectif = objectif.lower().strip()[0]
    print("Starting learning mode")
    time.sleep(5)
    return jsonify({"end": "None"})

@app.route("/video_feed")
def rec():
    return Response(capture_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_frames():
    global cap,objectif,result
    score = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        if objectif != "" and not result:
            with mp_hands.Hands(model_complexity=0,max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image,hand_landmarks,mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())

                    a=np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    a=normalizeData(a)
                    a = torch.tensor(a).float()
                    image = cv2.flip(image, 1)
                    output = handmodel(a)
                    print("Output: ", output)
                    global classes
                    objid = classes.index(objectif)
                    id = torch.argmax(output).item()
                    res = torch.max(output).item()
                    if id == objid and res > 0.85 and objid !=4:
                        print("Objectif atteint")
                        objectif = ""
                        result = True
                        score = (res-0.85)/(1-0.85)
        elif result and objectif == "":
            #afficher le score
            image = cv2.rectangle(image, (0,0), (int(cap.get(3)),int(cap.get(4)*0.1)), (0, 0, 0), -1)
            image = cv2.putText(image, "Score: "+str(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
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
