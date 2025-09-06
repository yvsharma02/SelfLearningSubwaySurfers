import grpc
import cv2
import numpy as np
from emulator_controller_pb2 import ImageFormat
import emulator_controller_pb2_grpc
import time

class Recorder:

    def __init__(self):
        options = [
            ('grpc.max_receive_message_length', 20 * 1024 * 1024)
        ]

        self.channel = grpc.insecure_channel("[::1]:8554", options=options)
        self.stub = emulator_controller_pb2_grpc.EmulatorControllerStub(self.channel)
        self.fmt = ImageFormat(format=ImageFormat.RGBA8888)

    def capture(self):
        frame = self.stub.getScreenshot(self.fmt)
        img = np.frombuffer(frame.image, dtype=np.uint8)
        img = img.reshape((frame.format.height, frame.format.width, 4))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img_bgr

    def stop(self):
        self.channel.close()