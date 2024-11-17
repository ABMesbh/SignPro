from flask import Flask, jsonify, redirect, render_template, request
import whisper
import soundfile as sf
import os
from datetime import datetime

# Load pre-trained Whisper model
model = whisper.load_model("small")

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


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
