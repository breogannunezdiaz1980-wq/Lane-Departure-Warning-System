import cv2 as cv
import numpy as np


class Filter_Lane_Departure:
    """
    This class represents the filter
    """
    
    def __init__(self, frame, filters_tuple):
        """
        Here we initialize the class:

        self.frame -> The original frame
        self.filters -> A tuple of filters
        """
        
        self.frame = frame
        self.filters = filters_tuple
    
    def roi_edge_filters(self):
        """
        This method convert BGR frame to gray scale, then detect the edges
        and apply the canvases
        """
        
        #BGR to gray scale
        frame_gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        #We blur the image in order to remove the noise
        if np.median(frame_gray) > 80:
            #Using the gradient we detect the edges
            frame_gray = cv.GaussianBlur(frame_gray, (9, 9), 0)
            canny_edge = cv.Canny(frame_gray, 30, 90)
        else:
            #Using the gradient we detect the edges
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            frame_gray = clahe.apply(frame_gray)
            frame_gray = cv.GaussianBlur(frame_gray, (7, 7), 0)
            canny_edge = cv.Canny(frame_gray, 10, 30)
        #We apply the draw triangle and the cleaner square to the frame
        roi_edge = cv.bitwise_and(canny_edge, self.filters[0])
        roi_edge = cv.bitwise_and(roi_edge, self.filters[2])
        return roi_edge
