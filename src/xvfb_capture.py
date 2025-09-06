from Xlib import display, X
import numpy as np
import cv2

class XvfbCapture:
    def __init__(self, display_name=":1"):
        self.disp = display.Display(display_name)
        self.root = self.disp.screen().root
        geom = self.root.get_geometry()
        self.width, self.height = geom.width, geom.height

    def capture(self):
        raw = self.root.get_image(
            0, 0, self.width, self.height,
            X.ZPixmap, 0xffffffff
        )
        frame = np.frombuffer(raw.data, dtype=np.uint8).reshape(
            (self.height, self.width, 4)
        )
        return frame[:, :, :3]
    
    def stop(self):
        pass