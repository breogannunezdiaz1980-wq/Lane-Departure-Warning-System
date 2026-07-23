import cv2 as cv
import numpy as np

cap = cv.VideoCapture('prueba1.MP4')
blank = np.zeros((1080, 1920), np.uint8)
points = np.array([[960, 570], [230, 800], [1640, 800]], dtype=np.int32)
cv.fillPoly(blank, [points], 255, lineType=cv.LINE_AA)
area = np.zeros((1080, 1920), np.uint8)
points = np.array([[960, 630], [650, 800], [1200, 800]], dtype=np.int32)
cv.fillPoly(area, [points], 255, lineType=cv.LINE_AA)

area_detect = np.zeros((1080, 1920), np.uint8)
points = np.array( [[1070, 455], [900, 455], [900, 800], [1070, 800]], dtype=np.int32)
blank2 = np.zeros((1080, 1920), np.uint8)
cv.fillPoly(blank2, [points], 255, lineType=cv.LINE_AA)
blank2 = cv.bitwise_not(blank2)
frame_count = 0
while True:
    state, frame = cap.read()
    if not state:
        print("Fin de la transmisión")
        break   
    frame = cv.resize(frame, (1920, 1080))
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.GaussianBlur(frame_gray, (9, 9), 0)
    canny_edge = cv.Canny(frame_gray, 30, 90)
    
    roi_edge = cv.bitwise_and(canny_edge, blank)
    lines = cv.HoughLinesP(roi_edge, rho=1, theta=np.pi/180, threshold=40, minLineLength=30, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2, = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2- y1) / (x2 - x1)
            if abs(slope) > 0.5:
                cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv.line(area_detect, (x1, y1), (x2, y2), 255, 3)
    
    detect = cv.bitwise_and(area_detect, area)
    detect = cv.bitwise_and(detect, blank2)
    cv.imshow('detector', detect)
    pixels = cv.countNonZero(detect)
    if pixels > 20: 
        frames_count += 1
        if frames_count > 6:
            print('A donde vas bobo? anda pa alla')
    else:
        frames_count = 0
    cv.imshow('Reproduccion Video', frame)
    area_detect[:, :] = 0
    
    if cv.waitKey(17) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()