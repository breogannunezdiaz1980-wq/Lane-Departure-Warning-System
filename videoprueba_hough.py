import cv2 as cv
import numpy as np


cap = cv.VideoCapture('prueba1.MP4')

#This framework is to draw in each frame
blank = np.zeros((1080, 1920), np.uint8)
points = np.array([[960, 570], [230, 800], [1640, 800]], dtype=np.int32)
cv.fillPoly(blank, [points], 255, lineType=cv.LINE_AA)

#This framework is to detect the lane departure.
area = np.zeros((1080, 1920), np.uint8)
points = np.array([[960, 630], [650, 800], [1200, 800]], dtype=np.int32)
cv.fillPoly(area, [points], 255, lineType=cv.LINE_AA)

#This framework is to clean the middle of the road in order to discard trash lines
points = np.array( [[1070, 455], [900, 455], [900, 800], [1070, 800]], dtype=np.int32)
blank2 = np.zeros((1080, 1920), np.uint8)
cv.fillPoly(blank2, [points], 255, lineType=cv.LINE_AA)
blank2 = cv.bitwise_not(blank2)

#This framework will have lines if the car is veering out of the lane. If the frame is black the car would be on the lane.
area_detect = np.zeros((1080, 1920), np.uint8)

#This frame counter is to prevent the false positives
frame_count = 0
#Here the video start
while True:
    state, frame = cap.read()
    if not state:
        print("end")
        break   
    #We use this to normalize the video
    frame = cv.resize(frame, (1920, 1080))

    #BGR to gray scale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #We blur the image in order to remove the noise
    frame_gray = cv.GaussianBlur(frame_gray, (9, 9), 0)
    #Using the gradient we detect the edges
    canny_edge = cv.Canny(frame_gray, 30, 90)

    #We apply the draw filter to the frame
    roi_edge = cv.bitwise_and(canny_edge, blank)
    #Using the Hough transform we draw the lines
    lines = cv.HoughLinesP(roi_edge, rho=1, theta=np.pi/180, threshold=40, minLineLength=30, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2, = line[0]
            #We discard the vertical lines
            if x2 - x1 == 0:
                continue
            #We discard the horizontal lines
            slope = (y2- y1) / (x2 - x1)
            #We use lines with slope > 0.5 to reduce nose
            if abs(slope) > 0.5:
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv.line(area_detect, (x1, y1), (x2, y2), 255, 3)

    #We apply the area detector filter
    detect = cv.bitwise_and(area_detect, area)
    #We clean the middle of the frame
    detect = cv.bitwise_and(detect, blank2)

    #This is only to see what the pc is watching
    cv.imshow('detector', detect)

    #We create a filter to discard the false positives
    pixels = cv.countNonZero(detect)
    if pixels > 20: 
        frame_count += 1
        if frame_count > 6:
            print('go back')
    else:
        #We have to restart the frame_count to avoid bugs
        frame_count = 0
    
    cv.imshow('Reproduccion Video', frame)
    #We clean the area detector
    area_detect[:, :] = 0

    #Press 'q' to close the windows
    if cv.waitKey(17) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
