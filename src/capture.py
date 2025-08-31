import grpc
import cv2
import numpy as np
from emulator_controller_pb2 import ImageFormat
import emulator_controller_pb2_grpc
import time

options = [
    ('grpc.max_receive_message_length', 20 * 1024 * 1024)
]

channel = grpc.insecure_channel("[::1]:8554", options=options)
stub = emulator_controller_pb2_grpc.EmulatorControllerStub(channel)

# Request frames in RGBA8888
fmt = ImageFormat(format=ImageFormat.RGBA8888)

c = 0
start_time = time.time()
while True:
    frame = stub.getScreenshot(fmt)  # single frame
    img = np.frombuffer(frame.image, dtype=np.uint8)
    img = img.reshape((frame.format.height, frame.format.width, 4))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(f"generated/Emulator_{c}.png", img_bgr)
    c += 1
    print(f"FPS: {c / (time.time() - start_time)}")

# c = 0
# # Stream frames
# for frame in stub.streamScreenshot(fmt):
#     print("Trying...")
#     img = np.frombuffer(frame.image, dtype=np.uint8)
#     img = img.reshape((frame.format.height, frame.format.width, 4))

#     # Convert RGBA → BGR for OpenCV
#     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

#     cv2.imwrite(f"generated/Emulator_{c}.png", img_bgr)
#     c += 1

# cv2.destroyAllWindows()
