//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var recorder; 						//WebAudioRecorder object
var input; 							//MediaStreamAudioSourceNode  we'll be recording
var encodingType; 					//holds selected encoding for resulting audio (file)
var encodeAfterRecord = true;       // when to encode

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext; //new audio context to help us record

var encodingTypeSelect = document.getElementById("encodingTypeSelect");
var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");

const webcamVideo = document.getElementById('webcamVideo');
const takePhotoButton = document.getElementById('take_letter_photo');

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

function startRecording() {
	console.log("startRecording() called");

	/*
		Simple constraints object, for more advanced features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    
    var constraints = { audio: true, video:false }

    /*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		// // __log("getUserMedia() success, stream created, initializing WebAudioRecorder...");

		/*
			create an audio context after getUserMedia is called
			sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
			the sampleRate defaults to the one set in your OS for your playback device

		*/
		audioContext = new AudioContext();

		//update the format 
		document.getElementById("formats").innerHTML="Format: 2 channel "+encodingTypeSelect.options[encodingTypeSelect.selectedIndex].value+" @ "+audioContext.sampleRate/1000+"kHz"

		//assign to gumStream for later use
		gumStream = stream;
		
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);
		
		//stop the input from playing back through the speakers
		//input.connect(audioContext.destination)

		//get the encoding 
		encodingType = encodingTypeSelect.options[encodingTypeSelect.selectedIndex].value;
		
		//disable the encoding selector
		encodingTypeSelect.disabled = true;

		recorder = new WebAudioRecorder(input, {
		  workerDir: "static/js/", // must end with slash
		  encoding: encodingType,
		  numChannels:2, //2 is the default, mp3 encoding supports only 2
		  onEncoderLoading: function(recorder, encoding) {
		    // show "loading encoder..." display
		    //  __log("Loading "+encoding+" encoder...");
		  },
		  onEncoderLoaded: function(recorder, encoding) {
		    // hide "loading encoder..." display
		    // __log(encoding+" encoder loaded");
		  }
		});

		recorder.onComplete = function(recorder, blob) { 
			// __log("Encoding complete");
			createDownloadLink(blob,recorder.encoding);
			encodingTypeSelect.disabled = false;
		}

		recorder.setOptions({
		  timeLimit:120,
		  encodeAfterRecord:encodeAfterRecord,
	      ogg: {quality: 0.5},
	      mp3: {bitRate: 160}
	    });

		//start the recording process
		recorder.startRecording();

		 // __log("Recording started");

	}).catch(function(err) {
	  	//enable the record button if getUSerMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;

	});

	//disable the record button
    recordButton.disabled = true;
    stopButton.disabled = false;
}

function stopRecording() {
	console.log("stopRecording() called");
	
	//stop microphone access
	gumStream.getAudioTracks()[0].stop();

	//disable the stop button
	stopButton.disabled = true;
	recordButton.disabled = false;
	
	//tell the recorder to finish the recording (stop recording + encode the recorded audio)
	recorder.finishRecording();

	// __log('Recording stopped');	
}

function createDownloadLink(blob,encoding) {
	
	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	// var li = document.createElement('li');
	var link = document.createElement('a');

	//add controls to the <audio> element
	au.controls = true;
	au.src = url;

	//link the a element to the blob
	link.href = url;
	link.download = new Date().toISOString() + '.'+encoding;
	link.innerHTML = link.download;
	// link.click();

	const formData = new FormData();
    formData.append("file", blob, "recording.wav");

	fetch("/", {
        method: "POST",
        body: formData
    })
	.then(response => response.json())
    .then(data => {
        // Display the transcription result on the page
        document.getElementById("speechText").value = data.transcription;
    })
    .catch(error => {
        console.error("Error uploading audio:", error);
    });
	console.log("Audio uploaded by post request");
	li.appendChild(au);
	
}

document.getElementById('captureButton').addEventListener('click', async () => {
	// webcamVideo.display = 'block';
	// takePhotoButton.display = 'block';
	const constraints = {
		video: {
			width: { min: 1280 },
			height: { min: 720 },
			facingMode: 'user'
		},
		audio: false
	};
	
	navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
		webcamVideo.srcObject = stream;
	
	}).catch((error) => {
		console.error('Error accessing media devices.', error);
	});

	takePhotoButton.addEventListener('click', () => {
		const canvas = document.createElement('canvas');
		const context = canvas.getContext('2d');
		canvas.width = webcamVideo.videoWidth;
		canvas.height = webcamVideo.videoHeight;
		context.drawImage(webcamVideo, 0, 0, webcamVideo.videoWidth, webcamVideo.videoHeight);
		canvas.toBlob(async (blob) => {
			const formData = new FormData();
			formData.append('image', blob, 'image.jpg');
			const response = await fetch('/capture', {
				method: 'POST',
				body: formData
			});
			const data = await response.json();
			if (data.error) {
				alert(data.error);
			} else {
				document.getElementById("speechText").value = data.letter;
				// webcamVideo.style.display = 'none';
				// takePhotoButton.style.display = 'none';

				// Stop the video stream to turn off the camera
				const stream = webcamVideo.srcObject;
				const tracks = stream.getTracks(); // Get all media tracks (video/audio)
				tracks.forEach((track) => track.stop()); // Stop each track
				webcamVideo.srcObject = null; // Clear the video source

			}
		});
	});
	
	

	// try {
	// 	const response = await fetch('/capture', {
	// 		method: 'POST'
	// 	});

	// 	const data = await response.json();
	// 	if (data.error) {
	// 		alert(data.error);
	// 	} else {
	// 		document.getElementById("speechText").value = data.letter;
	// 	}
	// } catch (error) {
	// 	console.error('Error:', error);
	// }
});

document.getElementById('startButton').addEventListener('click', async () => {
	const text = document.getElementById('speechText').value;
	const video_feed = document.getElementById('video_feed');
	const image = document.getElementById('image');
	const firstLetter = text.charAt(0).toLowerCase();
	image.src = `static/${firstLetter}.jpg`;
	video_feed.style.display = 'none';
	image.style.display = 'block';
	const formData = new FormData();
    formData.append("transcript", text);
	
	const response = await fetch('/start', {
		method: 'POST',
		body:formData

	});
	const data = await response.json();
	video_feed.style.display = 'block';
	image.style.display = 'none';
});