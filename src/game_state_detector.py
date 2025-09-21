import custom_enums
import cv2
import numpy as np
from collections import deque

class StateDetector:

    def __init__(self, track_size=4, detect_bound=0.41, undetect_bound=0.28):
        self.reference_pause = cv2.imread("data/reference_images/pause_button.png")
        self.reference_police = cv2.imread("data/reference_images/police4.png", cv2.IMREAD_UNCHANGED) #RGBA
        self.scaled_police = self.build_scaled_templates(self.reference_police)
        # self.reference_doggo = cv2.imread("data/reference_images/doggo2.png")

        self.detected_for = 0
        self.last_detected = False

        self.track_size=track_size
        self.detect_bound = detect_bound
        self.undetect_bound = undetect_bound
        self.police_state_queue = deque(maxlen=track_size)
        self.last_time_detected = None

    def build_scaled_templates(self, reference, scales=np.linspace(.4, 1.4, 50)):
        templates = []
        for scale in scales:
            new_w = int(reference.shape[1] * scale)
            new_h = int(reference.shape[0] * scale)
            if new_w < 10 or new_h < 10:  # skip too small
                continue
            resized = cv2.resize(reference, (new_w, new_h))
            templates.append(resized)
        
        return templates

    def detect_gamestate(self,  capture):
        x1, y1, x2, y2 = 11 / 4, 10 / 4, 66 / 4, 67 / 4
        result = cv2.matchTemplate(capture[x1:x2, y1:y2, :], self.reference_pause, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if (max_val > .4):
            return custom_enums.GAME_STATE_ONGOING
        
        return custom_enums.GAME_STATE_OVER
    
    def detector(self, capture, scaled_references):
        x1, y1, x2, y2 = 0, 60, 120, 200
        patch = capture[y1:y2, x1:x2]
        cv2.imwrite("patch.png", patch)
        true_max = 0
        for i, scaled in enumerate(scaled_references):
            if (scaled.shape[0] > patch.shape[0] or scaled.shape[1] > patch.shape[1]):
                continue
            cv2.imwrite(f"ref-{i}.png", scaled)
#            mask = cv2.threshold(scaled[:, :, 3], 0, 255, cv2.THRESH_BINARY)[1]
            result = cv2.matchTemplate(patch, scaled[:, :, 0:3], cv2.TM_CCOEFF_NORMED)#, mask=mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            true_max = max(max_val, true_max)

        return true_max

    def detect_police(self, capture):
        confidence = self.detector(capture, self.scaled_police)
        # print(confidence)
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