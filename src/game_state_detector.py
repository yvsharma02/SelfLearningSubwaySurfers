import custom_enums
import cv2
import numpy as np

class StateDetector:

    def __init__(self):
        self.reference_pause = cv2.imread("data/reference_images/pause_button.png")
        self.reference_pause = cv2.cvtColor(self.reference_pause, cv2.COLOR_BGR2RGB)

    def detect_gamestate(self,  capture):
        x1, y1, x2, y2 = 11, 10, 66, 67
        result = cv2.matchTemplate(self.reference_pause, capture[x1:x2, y1:y2, :], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if (max_val > .4):
            return custom_enums.GAME_STATE_ONGOING
        
        return custom_enums.GAME_STATE_OVER