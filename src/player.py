from save_queue import SaveQue
from grpc_controller import EmulatorController
import constants
from ppadb.client import Client as AdbClient
import time
import emulator_utils
import ssai_model
import game_state_detector
from PIL import Image
import trainer
import torch
from ingame_run import InGameRun
import cv2
import os
import shutil
from collections import deque
from grpc._channel import _InactiveRpcError
from collections import deque

NOTHING_SAMPLING_RATE_ONE_IN_X = 30
RETRAIN_AFTER_X_RUNS = 10

keypress_action_map = {
    "NONE": constants.ACTION_NOTHING,
    "KEY_UP": constants.ACTION_UP,
    "KEY_DOWN": constants.ACTION_DOWN,
    "KEY_LEFT": constants.ACTION_LEFT,
    "KEY_RIGHT": constants.ACTION_RIGHT,
}

class Player:
    def __init__(self, model=None, device=None):
        self.gsd = game_state_detector.StateDetector()
        self.controller = EmulatorController()
        self.model = model
        self.device = device
        self.current_run = None
        self.run_no = 0
        self.rgb_queue = deque([],maxlen=3)
        self.bgr_queue = deque([],maxlen=3)
        self.frame_number = 0
        self.last_frame_time = time.time()
        self.frame_time_tracker = deque(maxlen=60)
        self.total_time = 0
        self.total_frames = 0
        # self.input_controller = MultiDeviceReader()

    def get_dataset_len(self):
        x = len(os.listdir("generated/runs/dataset"))
        return x

    def start(self):
        if (self.current_run != None):
            return

        self.dataset = time.strftime(('%Y-%m-%d %H:%M:%S %Z'), time.localtime(time.time()))
        print(f"Starting Recording... Dataset: ${self.dataset}")
        self.frame_number = 0
        self.save_que = SaveQue(self.dataset, f"generated/runs/dataset/{self.dataset}")
        self.stream_path = f"generated/streams/{self.dataset}"
        self.current_run = InGameRun(self.controller, self.save_que)
        self.run_no += 1

        # os.makedirs(self.stream_path, exist_ok=True)

    def stop(self):
        if (self.current_run == None): 
            return

        print(f"Stopping Recording...: {self.run_no}")
        self.current_run.close()
        self.current_run = None
        if (self.run_no % 25 == 0):
            trainer.main()
            self.model, self.device = ssai_model.load("generated/models/test.pth")

    def tick(self):
        now = time.time()
        img_rgb = self.controller.capture()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.rgb_queue.append(img_rgb)
        self.bgr_queue.append(img_bgr)
        gamestate = self.gsd.detect_gamestate(img_bgr)
        lane = self.gsd.detect_lane(img_bgr)

        if (self.current_run == None and gamestate == constants.GAME_STATE_OVER):
            # cv2.imwrite("capture.png", img_bgr)
            # time.sleep(1)
            self.controller.tap(400, 750)
            self.controller.tap(432, 105)

        if (self.current_run == None and gamestate < constants.GAME_STATE_OVER):
            self.start()

        if (self.current_run != None):
            if (len(self.rgb_queue) >= 3):
                action, logits = self.model.infer([Image.fromarray(x) for x in self.rgb_queue], self.current_run.run_secs(), self.device, randomize=self.get_dataset_len() <= 25)

                if (gamestate != constants.GAME_STATE_OVER):
                    self.current_run.give_command(action, list(self.bgr_queue), gamestate, logits, lane, self.frame_number)

                self.current_run.tick(gamestate, lane)

            if (self.current_run.is_finished()):
                self.stop()
                self.current_run = None

            # if (now - self.last_frame_time >= 1.0 / 60.0):
#            cv2.imwrite(os.path.join(self.stream_path, f"{self.frame_number}.png"), img_bgr)
            self.frame_number += 1

        self.last_frame_time = now
        frame_time = time.time() - now
        # self.frame_time_tracker.append(frame_time)
        # frame_time_avg60 = sum(self.frame_time_tracker) / len(self.frame_time_tracker)
        # print(f"Frame Time: {((time.time() - now) * 1000.0):.2f}; last_60_avg: {(frame_time_avg60 * 1000.0):0.2f}")
        # print(f"FPS: {(1.0 / frame_time):.0f}, last_60_avg: {(1.0 / frame_time_avg60):.0f}")
        self.total_time += frame_time
        self.total_frames += 1
        avg_frame_time = self.total_time / self.total_frames
        if (self.total_frames % 1000 == 0):
            print(f"Converged: {(1.0 / avg_frame_time):.0f} FPS")
        # time.sleep(0.075)
        return True

    def start_mainloop(self):
        while True:
            try:
                if (not self.tick()):
                    break
            except KeyboardInterrupt as e:
                break
            except _InactiveRpcError:
                pass
        self.stop()


def main():
    # model, device = None, None
    model, device = ssai_model.load("generated/models/test.pth") if os.path.exists("generated/models/test.pth") else (ssai_model.SSAIModel(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    os.makedirs("generated", exist_ok=True)
    logfile = open("generated/emu_log.txt", "w+")
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    print("Launching Emulator. Please Wait. Should not take longer than 120s.")
    emulator_utils.launch(adb_client, 120, stderror=logfile, stdout=logfile)
    player = Player(model, device)
    player.start_mainloop()
    logfile.close()
    

if __name__ == "__main__":
    main()