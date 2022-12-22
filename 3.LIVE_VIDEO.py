import cv2
import numpy as np

# load the Haar cascade classifiers for the face, smile, and eyes
face_cascade = cv2.CascadeClassifier("opencvlib\haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("opencvlib\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("opencvlib\haarcascade_smile.xml")

# Create the kernel for the dilation and erosion operations
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

# create VideoCapture object to capture video from webcam
cap = cv2.VideoCapture(0)

# loop until user quits
while True:
  # capture frame-by-frame
  ret, frame = cap.read()

  # check if frame was successfully captured
  if ret == True:
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # use Haar cascade classifier to detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # loop through each detected face
    for (x, y, w, h) in faces:
      # extract the face ROI
      roi_gray = gray[y:y+h, x:x+w]

      # use Haar cascade classifier to detect smiles in the face ROI
      smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 5)

      # use Haar cascade classifier to detect eyes in the face ROI
      eyes = eyes_cascade.detectMultiScale(roi_gray, 1.3, 5)

      # if a smile is detected, but no eyes are detected,
      # it is likely that the person is lying
      if len(smiles) > 0 and len(eyes) == 0:
        prediction = 1
      else:
        prediction = 0

      # draw a rectangle around the face
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # if the person is predicted to be lying, display "LIE" on the frame
      if prediction == 1:
        cv2.putText(frame, "LIE", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Concatenate the original frame and the preprocessed frame side by side
    result = np.concatenate((frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)), axis=1)

    # Show the concatenated frames
    #cv2.imshow("Live video and preprocessed video", result)
    # display the frame
    cv2.imshow('Lie Detection', frame)

    # check if user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# release the VideoCapture object
cap.release()

# destroy all windows
cv2.destroyAllWindows()



