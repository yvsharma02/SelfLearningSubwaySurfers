import custom_enums
import cv2
import numpy as np
from collections import deque

class StateDetector:

    def __init__(self, track_size=10, threshold=5):
        self.reference_pause = cv2.imread("data/reference_images/pause_button.png")
        self.reference_police = cv2.imread("data/reference_images/police4.png")
        self.reference_doggo = cv2.imread("data/reference_images/doggo2.png")

        self.detected_for = 0
        self.last_detected = False

        self.track_size=track_size
        self.threshold=threshold
        self.police_state_queue = deque(maxlen=track_size)

    def detect_gamestate(self,  capture):
        x1, y1, x2, y2 = 11, 10, 66, 67
        result = cv2.matchTemplate(capture[x1:x2, y1:y2, :], self.reference_pause, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if (max_val > .4):
            return custom_enums.GAME_STATE_ONGOING
        
        return custom_enums.GAME_STATE_OVER
    
    def detector(self, capture, reference):
        # ROI crop
        x1, y1, x2, y2 = 0, 550, 480, 800
        patch = capture[y1:y2, x1:x2]

        # Template matching
        cv2.imwrite("ref.png", self.reference_police)
        cv2.imwrite("patch.png", patch)
        # cv2.waitKey()
        result = cv2.matchTemplate(patch, reference, cv2.TM_CCOEFF_NORMED)

        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # print("Best match score:", max_val)
        # print("Best location:", max_loc)

        return max_val
        # if max_val > 0.8:   # threshold (tune)
        #     return True
        # else:
        #     print("❌ Feature not found")


        # x1, y1, x2, y2 = 0, 400, 480, 800
        # patch = capture[y1:y2, x1:x2]   # crop ROI from capture

        # # Convert to grayscale (ORB works better on grayscale)
        # patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        # ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

        # # Initialize ORB
        # orb = cv2.ORB_create()

        # # Detect keypoints and descriptors
        # kp1, des1 = orb.detectAndCompute(ref_gray, None)
        # kp2, des2 = orb.detectAndCompute(patch_gray, None)

        # # detected = False  # default state

        # if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
        #     # Match descriptors using brute-force matcher
        #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        #     matches = bf.match(des1, des2)

        #     # Filter good matches (distance < threshold)
        #     good_matches = [m for m in matches if m.distance < 60]  # tune threshold
        #     # print(len(good_matches))
        #     # Decide detection based on number of good matches
        #     return len(good_matches) # tune threshold
        # else:
        #     return 0

        # # State change check
        # if self.last_detected is None or detected != self.last_detected:
        #     self.last_detected = detected
        #     print("Current State: " + ("detected" if detected else "not detected"))

    def detect_police(self, capture):
        # MIN_HOLD = 4
        THRESHOLD = .45
        police = self.detector(capture, self.reference_police)
        # doggo = self.detector(capture, self.reference_doggo)
        # print(f"police: ${police}, doggo: ${doggo}")
        self.police_state_queue.append(police)
        if (len(self.police_state_queue) < self.track_size):
            return False
        detected = False
        if (sum(self.police_state_queue) / len(self.police_state_queue) > THRESHOLD):
            detected = True
            # self.detected_for += 1
        # else:
            # self.detected_for = 0

        return detected
        # print(self.police_state_queue.count(True))
        # return self.police_state_queue.count(True) > 6

        # detected = False
        # if (self.detected_for >= MIN_HOLD):
        #     detected = True

        # if (self.last_detected == None or self.last_detected != detected):
        #     self.last_detected = detected
        #     print("Current State: " + ("detected" if detected else "not detected"))