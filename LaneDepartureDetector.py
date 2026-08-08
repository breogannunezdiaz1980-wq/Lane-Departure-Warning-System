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
    
    def __init__(self, route, frame_shape=(1080, 1920), type="car"):
        """
        Here we initialize a new detector with two arguments:
        
        self.route -> video route
        self.shape -> video dimensions
        """
        
        self.route = route
        self.shape = frame_shape
        self.type = type

    def canvases(self):
        """
        This method creates the necessary masks using scaling.
        
        draw_square -> It will be use by Filters class to clean the frame
        detector_squares -> It will br use by the Alert class
        cleaner_triangle -> It will be use by Filters (Actually, this does not work)
        """
        draw_square = np.zeros(self.shape, np.uint8)
        points = np.array(
                [[self.shape[1]//12, self.shape[0]], 
                [self.shape[1]//2.157303, self.shape[0]//1.5],
                [self.shape[1]//1.864077, self.shape[0]//1.5],
                [self.shape[1]//1.0909091, self.shape[0]]],
                dtype=np.int32)
        cv.fillPoly(draw_square, [points], 255, lineType=cv.LINE_AA)

        detector_square = np.zeros(self.shape, np.uint8)
        points = np.array(
                [[self.shape[1]//2.577181, self.shape[0]],
                 [self.shape[1]//2.245614, self.shape[0]//1.27],
                 [self.shape[1]//2.169492, self.shape[0]//1.27],
                 [self.shape[1]//2.327273, self.shape[0]]],
                dtype=np.int32)
        cv.fillPoly(detector_square, [points], 255, lineType=cv.LINE_AA)
        points = np.array(
                [[self.shape[1]//1.634043, self.shape[0]],
                [self.shape[1]//1.802817, self.shape[0]//1.27],
                [self.shape[1]//1.855072, self.shape[0]//1.27],
                [self.shape[1]//1.753425, self.shape[0]]],
                dtype=np.int32)
        cv.fillPoly(detector_square, [points], 255, lineType=cv.LINE_AA)

        cleaner_triangle = np.zeros(self.shape, np.uint8)
        points = np.array(
                [[self.shape[1]//4.26666667, self.shape[0]//1.8],
                [self.shape[1]//1.30612, self.shape[0]//1.8],
                [self.shape[1]//2, self.shape[0]]],
                dtype=np.int32)
        cv.fillPoly(cleaner_triangle, [points], 255, lineType=cv.LINE_AA)
        cleaner_triangle = cv.bitwise_not(cleaner_triangle)
        return (draw_square, detector_square, cleaner_triangle)

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
            hough = Hough(roi_edge, frame, self.shape)
            frame, area_detector = hough.hough_transform()
            #We see if the car is leaking out.
            alertt.lane_departure_detector(area_detector, frame)
            
            #We reproduce the video with the lines.
            cv.imshow('Reproduccion Video', frame)
            #Press 'q' to close the windows
            if cv.waitKey(33) & 0xFF == ord('q'):
                break
        self.route.release()
        cv.destroyAllWindows()
