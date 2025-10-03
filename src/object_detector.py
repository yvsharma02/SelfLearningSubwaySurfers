import cv2
import numpy as np
import constants

from collections import deque

class ObjDetector:
    def __init__(self, unscaled_reference, frame_window_size, detect_bound, undetect_bound, debounce_window, lowest_scale, highest_scale, scale_samples, unscaled_patch_area, log_confidence, detect_bound_upper_limit = 1.0):
        self.reference = cv2.imread(unscaled_reference)
        self.scaled_references = ObjDetector.build_scaled_templates(self.reference, scales=np.linspace(lowest_scale, highest_scale, scale_samples))

        self.detect_bound_upper_limit = detect_bound_upper_limit
        self.frame_window_size = frame_window_size
        self.detect_bound = detect_bound
        self.undetect_bound = undetect_bound
        self.police_state_queue = deque(maxlen=frame_window_size)
        self.debounce_window=debounce_window
        self.lowest_scale = lowest_scale
        self.highest_scale = highest_scale
        self.scale_samples = scale_samples
        self.unscaled_patch_area = unscaled_patch_area

        self.last_detected = False
        self.detected_for = 0
        self.log_confidence = log_confidence
    
    def detect_raw(self, capture):
        x1, y1, x2, y2 = constants.scale_dimensions(*self.unscaled_patch_area)
        patch = capture[y1:y2, x1:x2]
        true_max = 0
        # cv2.imwrite("patch.png", patch)

        for i, scaled in enumerate(self.scaled_references):
            if scaled.shape[0] > patch.shape[0] or scaled.shape[1] > patch.shape[1]:
                continue
            # cv2.imwrite(f"ref-{i}.png", scaled)
            if (scaled.shape[2] == 4):
                mask = cv2.threshold(scaled[:, :, 3], 0, 255, cv2.THRESH_BINARY)[1]
                result = cv2.matchTemplate(patch, scaled[:, :, :3], cv2.TM_CCORR_NORMED, mask=mask)
            else:
                result = cv2.matchTemplate(patch, scaled, cv2.TM_CCORR_NORMED)

            _, max_val, _, _ = cv2.minMaxLoc(result)
            true_max = max(max_val, true_max)

        return true_max

    def detect(self, capture):
        confidence = self.detect_raw(capture)
        self.police_state_queue.append(confidence)
        if (self.log_confidence):
            print(confidence)

        if len(self.police_state_queue) < self.frame_window_size:
            return False

        avg_confidence = ObjDetector.weighted_average(self.police_state_queue)

        new_detection = self.last_detected
        if avg_confidence > self.detect_bound and avg_confidence < self.detect_bound_upper_limit:
            if not self.last_detected:
                self.detected_for += 1
                if self.detected_for >= self.debounce_window:
                    new_detection = True
                    self.detected_for = 0
        elif avg_confidence < self.undetect_bound:
            if self.last_detected:
                self.detected_for += 1
                if self.detected_for >= self.debounce_window:
                    new_detection = False
                    self.detected_for = 0
        else:
            self.detected_for = 0

        self.last_detected = new_detection
        return new_detection


    def weighted_average(queue, alpha=0.6):
        avg = 0
        total_weight = 0
        for i, val in enumerate(queue):
            weight = alpha * (1 - alpha) ** (len(queue) - 1 - i)
            avg += val * weight
            total_weight += weight
        return avg / total_weight if total_weight > 0 else 0

    def build_scaled_templates(reference, scales):
        templates = []
        for scale in scales:
            new_w = int(reference.shape[1] * scale)
            new_h = int(reference.shape[0] * scale)
            if new_w < 10 or new_h < 10:
                continue
            resized = cv2.resize(reference, (new_w, new_h))
            templates.append(resized)
        return templates