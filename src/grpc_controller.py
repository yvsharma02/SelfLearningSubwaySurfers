import grpc
import time
import emulator_controller_pb2 as emu_pb2
import emulator_controller_pb2_grpc as emu_pb2_grpc
import numpy as np
import cv2
import gc
import constants
from PIL import Image

class EmulatorController:
    def __init__(self):
        options = [
            ('grpc.max_receive_message_length', 480 * 800 * 3 * 1.25)
        ]
        self.channel = grpc.insecure_channel("[::1]:8554", options=options)
        self.stub = emu_pb2_grpc.EmulatorControllerStub(self.channel)
        self.fmt = emu_pb2.ImageFormat(format=emu_pb2.ImageFormat.RGB888)


    def capture(self):
        frame = self.stub.getScreenshot(self.fmt)
        img = np.frombuffer(frame.image, dtype=np.uint8)
        img_scaled = cv2.resize (img.reshape((frame.format.height, frame.format.width, 3)), (constants.SCALED_WIDTH, constants.SCALED_HEIGHT), interpolation=cv2.INTER_NEAREST)
        return img_scaled

    def stop(self):
        self.channel.close()

    def tap(self, x, y):
        self.stub.sendTouch(emu_pb2.TouchEvent(touches=[emu_pb2.Touch(
            identifier=1,
            x=x,
            y=y,
            pressure=1
        )]))

        self.stub.sendTouch(emu_pb2.TouchEvent(touches=[emu_pb2.Touch(
            identifier=1,
            x=x,
            y=y,
            pressure=0
        )]))

    def swipe(self, x_offset, y_offset, x_start=None, y_start=None, steps=10):
        if x_start is None:
            x_start = 300
        if y_start is None:
            y_start = 300

        for step in range(steps):
            x = x_start + int(x_offset * step / steps)
            y = y_start + int(y_offset * step / steps)
            pressure = 1 if step < steps - 1 else 0
            touch = emu_pb2.Touch(identifier=0, x=x, y=y, pressure=pressure)
            self.stub.sendTouch(emu_pb2.TouchEvent(touches=[touch]))


    def swipe_right(self):
        self.swipe(x_offset=100, y_offset=0)

    def swipe_left(self):
        self.swipe(x_offset=-100, y_offset=0)

    def swipe_down(self):
        self.swipe(x_offset=0, y_offset=100)

    def swipe_up(self):
        self.swipe(x_offset=0, y_offset=-100)
