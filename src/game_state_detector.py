import constants
import cv2
import numpy as np
from collections import deque

class StateDetector:

    def __init__(self, track_size=3, detect_bound=0.98, undetect_bound=0.96):
        self.reference_pause = cv2.imread("data/reference_images/pause_button.png")
        self.reference_police = cv2.imread("data/reference_images/police4.png", cv2.IMREAD_UNCHANGED)  # RGBA
        self.scaled_police = self.build_scaled_templates(self.reference_police)

        self.track_size = track_size
        self.detect_bound = detect_bound
        self.undetect_bound = undetect_bound
        self.police_state_queue = deque(maxlen=track_size)

        self.last_detected = False
        self.detected_for = 0

    def build_scaled_templates(self, reference, scales=np.linspace(.05, 1, 20)):
        templates = []
        for scale in scales:
            new_w = int(reference.shape[1] * scale)
            new_h = int(reference.shape[0] * scale)
            if new_w < 10 or new_h < 10:
                continue
            resized = cv2.resize(reference, (new_w, new_h))
            templates.append(resized)
        return templates

    def detector(self, capture, scaled_references):
        x1, y1, x2, y2 = constants.scale_dimensions(0, 360, 480, 800)
        patch = capture[y1:y2, x1:x2]
        true_max = 0

        for scaled in scaled_references:
            if scaled.shape[0] > patch.shape[0] or scaled.shape[1] > patch.shape[1]:
                continue
            mask = cv2.threshold(scaled[:, :, 3], 0, 255, cv2.THRESH_BINARY)[1]
            result = cv2.matchTemplate(patch, scaled[:, :, :3], cv2.TM_CCORR_NORMED, mask=mask)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            true_max = max(max_val, true_max)

        return true_max

    def weighted_average(self, queue, alpha=0.6):
        avg = 0
        total_weight = 0
        for i, val in enumerate(queue):
            weight = alpha * (1 - alpha) ** (len(queue) - 1 - i)
            avg += val * weight
            total_weight += weight
        return avg / total_weight if total_weight > 0 else 0

    def detect_police(self, capture):
        confidence = self.detector(capture, self.scaled_police)
        self.police_state_queue.append(confidence)
        # print(confidence)

        if len(self.police_state_queue) < self.track_size:
            return False

        avg_confidence = self.weighted_average(self.police_state_queue)

        new_detection = self.last_detected
        if avg_confidence > self.detect_bound:
            if not self.last_detected:
                self.detected_for += 1
                if self.detected_for >= self.track_size:
                    new_detection = True
                    self.detected_for = 0
        elif avg_confidence < self.undetect_bound:
            if self.last_detected:
                self.detected_for += 1
                if self.detected_for >= self.track_size:
                    new_detection = False
                    self.detected_for = 0
        else:
            self.detected_for = 0

        self.last_detected = new_detection
        return new_detection
