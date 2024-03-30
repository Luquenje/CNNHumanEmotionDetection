import sys
import cv2
import numpy as np
from keras.models import model_from_json
import os

use_webcam = False
default_video_path = 'sample2.mp4'

# total cmd line arguments
n = len(sys.argv)
if (n == 2):
    if sys.argv[1] == "Webcam":
        use_webcam = True
    else:
        default_video_path = sys.argv[1]

if os.path.isfile(default_video_path) == False:
    print("File does not exist")
    quit()

# filepath
# training_data_filepath = 'human_emotion_training_data/train'
# validation_data_filepath = 'human_emotion_training_data/test'

# fl = os.listdir(validation_data_filepath)

# print(fl)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

print(emotion_dict)

# load json and create model
json_file = open('models/ResNet50_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("models/ResNet50_model.h5")
print("Loaded model from disk")

# Load the video
video_path = default_video_path
cap = cv2.VideoCapture(video_path)

if use_webcam:
    # start the webcam feed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Get video details
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Set the desired height for display
desired_height = 800  # Adjust this value according to your screen resolution
desired_width = int((desired_height / frame_height) * frame_width)

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Process the video
while True:
    ret, frame = cap.read()
    
    if use_webcam == False:
        frame = cv2.resize(frame, (desired_width, desired_height))
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray_frame = frame# cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Resize the face ROI to 48x48 (same size as the FER2013 dataset)
        face_roi = cv2.resize(face_roi, (224, 224))
        
        # Normalize the face ROI
        face_roi = face_roi / 255.0
        
        # Reshape the face ROI
        face_roi = np.reshape(face_roi, (1, 224, 224, 3))
        
        # Predict the emotion
        prediction = emotion_model.predict(face_roi)
        
        # Get the predicted emotion label
        emotion_label = emotion_dict[np.argmax(prediction)]
        
        # Draw the emotion label on the frame
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any open CV windows
cap.release()
cv2.destroyAllWindows()



# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1280, 720))
#     if not ret:
#         break
    
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(gray_frame)

#     # take each face available on the camera and Preprocess it
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     cv2.imshow('Emotion Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()