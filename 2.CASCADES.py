import cv2
import numpy as np

# UNCOMMENT THE IMAGE OF THE PERSON WE CHOOSE
#original = cv2.imread('alex_gray140.jpg')
#original = cv2.imread('pavlos_gray0.jpg')
original = cv2.imread('ektor_gray140.jpg')
#original = cv2.imread('milos_gray70.jpg')

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
img = cv2.resize(original, (0,0), fx= 0.4, fy= 0.4)
copy = img.copy()
#print(img.shape)

# HAAR CASCADES

# Loads detectors from local opencv library
detectors = {
   "face": "opencvlib/haarcascade_frontalface_default.xml",
   "eyes": "opencvlib/haarcascade_eye.xml",
   "smile": "opencvlib/haarcascade_smile.xml",
    
}

# Classifies the path file as a cascade
facedetect = cv2.CascadeClassifier(detectors['face'])
eyedetect = cv2.CascadeClassifier(detectors['eyes'])
smiledetect = cv2.CascadeClassifier(detectors['smile'])

# Face detection
face = facedetect.detectMultiScale(
		img, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

# UNCOMMENT FOR EACH PERSON

# ALEX
#minNeighbors_EYE = 10
# PAVLOS
#minNeighbors_EYE = 10
# EKTOR
minNeighbors_EYE = 6
# MILOS
#minNeighbors_EYE = 10

#ALEX
#scaleFactor_EYE = 1.1
# PAVLOS
#scaleFactor_EYE = 1.1
# EKTOR
scaleFactor_EYE = 1.1
# MILOS
#scaleFactor_EYE = 1.1

# ALEX
#minNeighbors_SMILE = 15
# PAVLOS
#minNeighbors_SMILE = 10
# EKTOR
minNeighbors_SMILE = 10
# MILOS
#minNeighbors_SMILE = 10

#ALEX
#scaleFactor_SMILE = 1.1
# PAVLOS
#scaleFactor_SMILE = 1.1
# EKTOR
scaleFactor_SMILE = 1.1
# MILOS
#scaleFactor_SMILE = 1.1


# loop over the face bounding boxes
for (fX, fY, fW, fH) in face:
	# extract the face area
	faceArea = img[fY:fY+ fH, fX:fX + fW]
	# apply eyes detection to the face Area
	upper_half_faceArea = img[fY:fY+ int(fH/2), fX:fX + fW]
	eyeBox = eyedetect.detectMultiScale(
		upper_half_faceArea, scaleFactor=scaleFactor_EYE, minNeighbors=minNeighbors_EYE,
		minSize=(15, 15), maxSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
	# apply smile detection to the face Area
	bottom_half_faceArea = img[fY+int(fH/2):fY+fH, fX:fX + fW]
	smileBox = smiledetect.detectMultiScale(
		bottom_half_faceArea, scaleFactor=scaleFactor_SMILE, minNeighbors=minNeighbors_SMILE,
		minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)  
    # loop over the eye bounding boxes
	for (eX, eY, eW, eH) in eyeBox:
		# draw the eye bounding box
		eyeA = (fX + eX, fY + eY)
		eyeB = (fX + eX + eW, fY + eY + eH)
		cv2.rectangle(img, eyeA, eyeB, (0, 0, 255), 2)
        #ret, thresh1 = cv2.threshold(eyeBox,127,255,cv2.THRESH_BINARY)


	# loop over the smile bounding boxes
	for (sX, sY, sW, sH) in smileBox:
		# draw the smile bounding box
		ptA = (fX + sX, fY + int(fH/2)+ sY)
		ptB = (fX + sX + sW, fY + int(fH/2) + sY + sH)
		cv2.rectangle(img, ptA, ptB, (255, 0, 0), 2)
	# draw the face bounding box on the frame
	cv2.rectangle(img, (fX, fY), (fX + fW, fY + fH),(0, 255, 0), 2)


eyes = []
for i in eyeBox:
    crop_img = copy[fY + i[1]:fY + i[1]+ i[3], fX + i[0]:fX + i[0] + i[2]]
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_crop,127,255,cv2.THRESH_BINARY_INV)
    white = np.sum(thresh1 == 255)
    eyes.append(white)
    white_percentage = white/thresh1.size
    #print("The white pixels are: ", white)
    #print("The percentage of white pixels: ", "{:.0%}".format(white_percentage))
    #cv2.imshow("cropped", thresh1)
    #cv2.waitKey(0)


for i in smileBox:
    crop_img = copy[fY + int(fH/2) + i[1]:fY + int(fH/2) + i[1]+ i[3], fX + i[0]:fX + i[0] + i[2]]
    smile_height = i[3]
    #print("The smile's height is: ", smile_height)
    size = crop_img.size
    #print("The smile size is: ", size)
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_crop,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow("cropped", thresh1)
    #cv2.waitKey(0)



#cv2.imshow('image', img)
        
#cv2.waitKey(0)


# UNCOMMENT THE IMAGE OF THE PERSON WE CHOOSE
#original = cv2.imread('alex_gray70.jpg')
#original = cv2.imread('pavlos_gray70.jpg')
original = cv2.imread('ektor_gray70.jpg')
#original = cv2.imread('milos_gray140.jpg')

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
img2 = cv2.resize(original, (0,0), fx= 0.4, fy= 0.4)
copy = img2.copy()
#print(img2.shape)

# HAAR CASCADES

# Loads detectors from local opencv library
detectors = {
   "face": "opencvlib/haarcascade_frontalface_default.xml",
   "eyes": "opencvlib/haarcascade_eye.xml",
   "smile": "opencvlib/haarcascade_smile.xml",
    
}

# Classifies the path file as a cascade
facedetect = cv2.CascadeClassifier(detectors['face'])
eyedetect = cv2.CascadeClassifier(detectors['eyes'])
smiledetect = cv2.CascadeClassifier(detectors['smile'])

# Face detection
face = facedetect.detectMultiScale(
		img2, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

# UNCOMMENT FOR EACH PERSON

# ALEX
#minNeighbors_EYE = 10
# PAVLOS
#minNeighbors_EYE = 5
# EKTOR
minNeighbors_EYE = 10
# MILOS
#minNeighbors_EYE = 3

#ALEX
#scaleFactor_EYE = 1.1
# PAVLOS
#scaleFactor_EYE = 1.1
# EKTOR
scaleFactor_EYE = 1.1
# MILOS
#scaleFactor_EYE = 1.1

# ALEX
#minNeighbors_SMILE = 10
# PAVLOS
#minNeighbors_SMILE = 10
# EKTOR
minNeighbors_SMILE = 10
# MILOS
#minNeighbors_SMILE = 10

#ALEX
#scaleFactor_SMILE = 1.1
# PAVLOS
#scaleFactor_SMILE = 1.1
# EKTOR
scaleFactor_SMILE = 1.1
# MILOS
#scaleFactor_SMILE = 1.6


# loop over the face bounding boxes
for (fX, fY, fW, fH) in face:
	# extract the face area
	faceArea = img2[fY:fY+ fH, fX:fX + fW]
	# apply eyes detection to the face Area
	upper_half_faceArea = img2[fY:fY+ int(fH/2), fX:fX + fW]
	eyeBox = eyedetect.detectMultiScale(
		upper_half_faceArea, scaleFactor=scaleFactor_EYE, minNeighbors=minNeighbors_EYE,
		minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
	# apply smile detection to the face Area
	bottom_half_faceArea = img2[fY+int(fH/2):fY+fH, fX:fX + fW]
	smileBox = smiledetect.detectMultiScale(
		bottom_half_faceArea, scaleFactor=scaleFactor_SMILE, minNeighbors=minNeighbors_SMILE,
		minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)  
    # loop over the eye bounding boxes
	for (eX, eY, eW, eH) in eyeBox:
		# draw the eye bounding box
		eyeA = (fX + eX, fY + eY)
		eyeB = (fX + eX + eW, fY + eY + eH)
		cv2.rectangle(img2, eyeA, eyeB, (0, 0, 255), 2)
        #ret, thresh1 = cv2.threshold(eyeBox,127,255,cv2.THRESH_BINARY)


	# loop over the smile bounding boxes
	for (sX, sY, sW, sH) in smileBox:
		# draw the smile bounding box
		ptA = (fX + sX, fY + int(fH/2)+ sY)
		ptB = (fX + sX + sW, fY + int(fH/2) + sY + sH)
		cv2.rectangle(img2, ptA, ptB, (255, 0, 0), 2)
	# draw the face bounding box on the frame
	cv2.rectangle(img2, (fX, fY), (fX + fW, fY + fH),(0, 255, 0), 2)


eyes2=[]
for i in eyeBox:
    crop_img = copy[fY + i[1]:fY + i[1]+ i[3], fX + i[0]:fX + i[0] + i[2]]
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_crop,127,255,cv2.THRESH_BINARY_INV)
    white = np.sum(thresh1 == 255)
    eyes2.append(white)
    white_percentage = white/thresh1.size
    #print("The white pixels are: ", white)
    #print("The percentage of white pixels: ", "{:.0%}".format(white_percentage))
    #cv2.imshow("cropped", thresh1)
    #cv2.waitKey(0)

if len(eyes) == 2 and len(eyes2) == 1:
    eyes2.append(0)
if len(eyes2) == 2 and len(eyes) == 1:
    eyes.append(0)

for i in smileBox:
    crop_img = copy[fY + int(fH/2) + i[1]:fY + int(fH/2) + i[1]+ i[3], fX + i[0]:fX + i[0] + i[2]]
    smile_height = i[3]
    #print("The smile's height is: ", smile_height)
    size2 = crop_img.size
    #print("The smile size is: ", size2)
    gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray_crop,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow("cropped", thresh1)
    #cv2.waitKey(0)


#cv2.imshow('image2', img2)
#cv2.waitKey(0)
#print("For Ektor:")
#print("\n")
if size2>size:
    print('when the person is lying his smile is smaller')
else:
    print('when the person is lying his smile is bigger')

#ASSUMPTION: BOTH EYES ARE THE SAME

if eyes[0]>eyes2[0]:
    print('when the person is lying his eyes are more open')
else:
    print('when the person is lying his eyes are less open/squinting')




together = np.hstack((img, img2))
cv2.imshow('together', together)

cv2.waitKey(0)


cv2.destroyAllWindows()