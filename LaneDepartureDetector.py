import cv2 as cv
import numpy as np
from Filters import *
from Hough import *
from Alert import *

class Lane_Departure_Detector:
    """
    It represents the lane departure detector thar manages all the processes.
    
    This class provides a method that creates the necessary masks and another 
    method that runs the video.
    """
    
    def __init__(self, route, frame_shape=(1080, 1920)):
        """
        Here we initialize a new detector with two arguments:
        
        self.route -> video route
        self.shape -> video dimensions
        """
        
        self.route = route
        self.shape = frame_shape
    
    def canvases(self):
        """
        This method creates the necessary masks using scaling.
        
        draw_triangle -> It will be use by Filters class to clean the frame
        detector_triangle -> It will br use by the Alert class
        cleaner_ssquare -> It will be use by Filters (Actually, this does not work)
        """
        
        draw_triangle = np.zeros(self.shape, np.uint8)
        points = np.array(
            [[self.shape[1]//2, self.shape[0]//1.894736842], 
             [self.shape[1]//8.347826087, self.shape[0]//1.35],
             [self.shape[1]//1.170731707, self.shape[0]//1.35]],
            dtype=np.int32)
        cv.fillPoly(draw_triangle, [points], 255, lineType=cv.LINE_AA)

        detector_triangle = np.zeros(self.shape, np.uint8)
        points = np.array(
            [[self.shape[1]//2, self.shape[0]//1.714285714],
             [self.shape[1]//2.953846154, self.shape[0]//1.35],
             [self.shape[1]//1.6, self.shape[0]//1.35]],
            dtype=np.int32
        )
        cv.fillPoly(detector_triangle, [points], 255, lineType=cv.LINE_AA)
        
        cleaner_square = np.zeros(self.shape, np.uint8)
        points = np.array(
                    [[self.shape[1]//1.794392523, self.shape[0]//2.373626374],
                     [self.shape[1]//2.133333333, self.shape[0]//2.373626374],
                     [self.shape[1]//2.133333333, self.shape[0]//1.35],
                     [self.shape[1]//1.794392523, self.shape[0]//1.35]],
                    dtype=np.int32
                )
        cv.fillPoly(cleaner_square, [points], 255, lineType=cv.LINE_AA)
        cleaner_square = cv.bitwise_not(cleaner_square)
        
        return (draw_triangle, detector_triangle, cleaner_square)
    
    def runner(self):
        """
        This method runs the video and contains the other classes.

        1: Initialize the alert
        2: We run the video
        3: While video is running we clean, we draw and we detect where is the car

        If you want to close the window you have to press 'q'
        """

        #We create the canvases.
        filters_canvases = self.canvases()
        #We initialize the alert in order to detect the lane departure.
        alertt = Alert(filters_canvases[1])
        
        while True:
            state, frame = self.route.read()
            if not state:
                    print("end")
                    break
            
            #We use this to normalize the video.
            frame = cv.resize(frame, (self.shape[1], self.shape[0]))
            
            #We clean the image to do the Huge transform.
            filters = Filter_Lane_Departure(frame, filters_canvases)
            roi_edge = filters.roi_edge_filters()
            #We do the transform
            hough = Hough(roi_edge, frame)
            frame, area_detector = hough.hough_transform()
            #We see if the car is leaking out.
            alertt.lane_departure_detector(area_detector)
            
            #We reproduce the video with the lines.
            cv.imshow('Reproduccion Video', frame)
            #Press 'q' to close the windows
            if cv.waitKey(17) & 0xFF == ord('q'):
                break
        self.route.release()
        cv.destroyAllWindows()
