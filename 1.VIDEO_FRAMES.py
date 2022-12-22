import cv2

# IN ORDER TO GET THE OTHER THREE PERSON'S FRAMES
# YOU NEED TO UNCOMMENT THE LINES BELOW AND INSIDE THE WHILE LOOP

#cap = cv2.VideoCapture('PAVLOS1.mp4', apiPreference=cv2.CAP_MSMF)
cap = cv2.VideoCapture('MILOS1.mp4', apiPreference=cv2.CAP_MSMF)
#cap = cv2.VideoCapture('EKTOR1.mp4', apiPreference=cv2.CAP_MSMF)
#cap = cv2.VideoCapture('ALEX1.mp4', apiPreference=cv2.CAP_MSMF)

fgbg = cv2.createBackgroundSubtractorMOG2()

# COUNT THE FRAMES OF THE VIDEO
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS) 

# CALCULATE THE DURATION OF THE VIDEO
duration = frame_count / fps

count = 0
while(1):
    # DEPENDING ON THE VIDEO FRAME COUNT, WE CHOOSE THE FRAMES
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # WE DO THIS TO GET GRAYSCALED EVERY 70TH FRAME
    if ret:
        # UNCOMMENT THE PERSON YOU CHOOSE
        #cv2.imwrite('pavlos_gray{:d}.jpg'.format(count), gray)
        cv2.imwrite('milos_gray{:d}.jpg'.format(count), gray)
        #cv2.imwrite('ektor_gray{:d}.jpg'.format(count), gray)
        #cv2.imwrite('alex_gray{:d}.jpg'.format(count), gray)

        count += 70 # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        #cap.release()
        break
    
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    # WE DO THIS TO GET WITHOUT BACKGROUND EVERY 70TH FRAME
    # WE DON'T USE IT IN THE NEXT CODE, BUT WE MAY NEED IT LATER
    if ret:
        #cv2.imwrite('pavlos_masked{:d}.jpg'.format(count), fgmask)
        cv2.imwrite('milos_masked{:d}.jpg'.format(count), fgmask)
        #cv2.imwrite('ektor_masked{:d}.jpg'.format(count), fgmask)
        #cv2.imwrite('alex_masked{:d}.jpg'.format(count), fgmask)        
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        break
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
