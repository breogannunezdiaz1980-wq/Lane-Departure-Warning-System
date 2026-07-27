import cv2 as cv
import numpy as np

class Alert:
    """
    This class represents the alert of lane departure
    """
    
    def __init__(self, filter):
        """
        Here wue initialize the object:

        self.filter -> detector filter
        self.frame_count -> a frame counter to prevent false positives.
        """
        
        self.filter = filter
        self.frame_count = 0
    
    def lane_departure_detector(self, area_detector):
        """
        This method verify if the frame has a line
        """
        
        #We applay the detector filter
        detector = cv.bitwise_and(area_detector, self.filter)
        #We create a filter to discard the false positives
        pixels = cv.countNonZero(detector)
        if pixels > 20: 
            self.frame_count += 1
            if self.frame_count > 6:
                print('go back')
        else:
            #We have to restart the frame_count to avoid bugs
            self.frame_count = 0
