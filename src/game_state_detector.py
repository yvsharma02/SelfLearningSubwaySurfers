import constants
import cv2
import numpy as np
from collections import deque
from object_detector import ObjDetector

class StateDetector:

    def __init__(self):
        self.police_detector = ObjDetector(
            "data/reference_images/police4.png",
            3,
            0.98,
            0.96,
            3,
            0.05,
            1,
            20,
            (0, 360, 480, 800), log_confidence=False)
        
        self.pause_button_detector = ObjDetector(
            "data/reference_images/pause_button.png",
            1,
            0.86,
            0.86,
            0,
            0.05,
            1,
            20,
            (16, 12, 68, 64), log_confidence=False, detect_bound_upper_limit=0.866)

    def detect_gamestate(self, capture):
        if not self.pause_button_detector.detect(capture):
            return constants.GAME_STATE_OVER
        
        # if self.police_detector.detect(capture):
        #     return constants.GAME_STATE_NON_FATAL_MISTAKE
        
        return constants.GAME_STATE_ONGOING
        
    #     # return constants.GAME_STATE_ONGOING if pause_button_detected else constants.GAME_STATE_OVER        

    # def detect_police(self, capture):
    #     return self.police_detector.detect(capture)