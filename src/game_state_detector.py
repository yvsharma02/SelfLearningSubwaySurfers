import constants
import cv2
import numpy as np
from collections import deque
from object_detector import ObjDetector
import player_detector

class StateDetector:

    def __init__(self):
        self.lane_window = deque([], maxlen=4)
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

    def detect_lane(self, capture):
        new_lane = self.detect_lane_raw(capture)
        if (new_lane != -1):
            self.lane_window.append(new_lane)
        counts = [0, 0, 0]
        for x in self.lane_window:
            counts[x] += 1

        max_idx = 0
        for i in range(0, 3):
            if (counts[i] > counts[max_idx]):
                max_idx = i

        return max_idx

    def detect_lane_raw(self, capture):
        x, y = player_detector.get_blob_average_position(capture)
        if (x == None): return -1
        if (x < .4): return 0
        if (x > .58): return 2
        return 1

    def detect_gamestate(self, capture):
        if not self.pause_button_detector.detect(capture):
            return constants.GAME_STATE_OVER
        
        return constants.GAME_STATE_ONGOING
    