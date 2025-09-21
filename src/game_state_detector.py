import custom_enums
import cv2
import numpy as np
from collections import deque

class StateDetector:

    def __init__(self, track_size=6, detect_bound=0.5, undetect_bound=0.32):
        self.reference_pause = cv2.imread("data/reference_images/pause_button.png")
        self.reference_police = cv2.imread("data/reference_images/police4.png")
        self.reference_doggo = cv2.imread("data/reference_images/doggo2.png")

        self.detected_for = 0
        self.last_detected = False

        self.track_size=track_size
        self.detect_bound = detect_bound
        self.undetect_bound = undetect_bound
        self.police_state_queue = deque(maxlen=track_size)
        self.last_time_detected = None

    def detect_gamestate(self,  capture):
        x1, y1, x2, y2 = 11, 10, 66, 67
        result = cv2.matchTemplate(capture[x1:x2, y1:y2, :], self.reference_pause, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if (max_val > .4):
            return custom_enums.GAME_STATE_ONGOING
        
        return custom_enums.GAME_STATE_OVER
    
    def detector(self, capture, reference):
        x1, y1, x2, y2 = 0, 550, 480, 800
        patch = capture[y1:y2, x1:x2]
        result = cv2.matchTemplate(patch, reference, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        return max_val

    def detect_police(self, capture):
        confidence = self.detector(capture, self.reference_police)
        self.police_state_queue.append(confidence)

        if (len(self.police_state_queue) < self.track_size):
            return False
        
        avg_confidence = sum(self.police_state_queue) / len(self.police_state_queue)

        new_detection = None
        if (avg_confidence > self.detect_bound):
            new_detection = True
        if (avg_confidence < self.undetect_bound):
            new_detection = False

        if (new_detection is None):
            new_detection = self.last_detected
        
        self.last_detected = new_detection

        return new_detection