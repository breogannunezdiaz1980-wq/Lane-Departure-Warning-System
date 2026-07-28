import cv2 as cv
import numpy as np

class Hough:
    """
    This class represents the Hough transform.
    """
    
    def __init__(self, roi_edge, frame):
        """
        Here we initialize the object.

        self.roi_edge -> It is the frame with the filters.
        self.roi_edge -> It is the original frame.
        """
        self.roi_edge = roi_edge
        self.frame = frame
        
    def hough_transform(self):
        """
        This method draw the lines in the original frame and in another new one
        that will be use by Alert class.
        """

        #The new image
        area_detect = np.zeros((1080, 1920), np.uint8)
        #Using the Hough transform we draw the lines
        lines = cv.HoughLinesP(self.roi_edge, rho=1, theta=np.pi/180, threshold=40, minLineLength=30, maxLineGap=20)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2, = line
                #We discard the vertical lines
                if x2 - x1 == 0:
                    continue
                #We discard the horizontal lines
                slope = (y2- y1) / (x2 - x1)
                #We use lines with slope > 0.5 to reduce nose
                if abs(slope) > 0.5:
                    cv.line(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv.line(area_detect, (x1, y1), (x2, y2), 255, 3)
        return (self.frame, area_detect)
