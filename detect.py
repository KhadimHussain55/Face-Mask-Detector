import os
import cv2
import numpy as np
import datetime

#silence Tensorflow Logs
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

model =tf.keras.models.load_model("MyModel") 
label = [1,0]

casc_path = "frontalfacedefault.xml"
face_cascade = cv2.CascadeClassifier(casc_path)
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)
video_capture.set(10, 100)
if video_capture.isOpened:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
       
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            img = cv2.resize(roi_color, (200,200))
            img = np.reshape(img, (1,200,200,3))/255.0
            pred = np.argmax(model.predict(img))
            lbl = label[pred]
            print(lbl)
            if lbl==1:

                cv2.putText(frame,"Mask Detected",(x, y), font, 1,(255,0,0),2)
            else:
                cv2.putText(frame,"No Mask Detected",(x, y), font, 1,(255,0,0),2)
        #Displaying Exit Info
        cv2.rectangle(frame, (0,0), (180,24), (0,0,0), cv2.FILLED)
        cv2.putText(frame,"Press Q To Quit",(2, 20), font, 0.7,(0,230,0),2)
        #Displaying Date and Time
        cv2.putText(frame,str(datetime.datetime.now()),(273, 477), font, 1,(0,255,0),2)
        # Display the resulting frame
        cv2.imshow('Face Mask Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("Camera Not Detected")
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
