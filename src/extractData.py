import cv2
import mediapipe as mp
import numpy as np
from dataUtils import normalizeData
import os

cap = cv2.VideoCapture(0)
filename=str(input("name of the txt file :"))
root_path = os.path.dirname(os.path.abspath(__file__))
# Join the root path with the 'uploads' folder and the file's name
filepath = os.path.join(root_path, "uploads", filename+".txt")
file=open(filepath,"a")
nb=0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(model_complexity=0,max_num_hands=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
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
        k = cv2.waitKey(1)
        if k%256 == 27:
            cap.release()
            file.close()
            print("{0} data created in file {1}".format(nb,filename))
        while k%256 == 32:
            nb+=1
            a=np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            a=normalizeData(a)
            file.write(",".join([str(x) for x in a])+"\n")
            k = cv2.waitKey(5)
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))