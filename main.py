import cv2 as cv
import numpy as np
from LaneDeparture import *

if __name__ == "__main__":
    cap = cv.VideoCapture('prueba1.MP4')
    lane_departue = Lane_Departure_Detector(cap)
    lane_departue.runner()
