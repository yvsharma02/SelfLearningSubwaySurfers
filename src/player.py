from save_queue import SaveQue
from grpc_controller import EmulatorController
import constants
from ppadb.client import Client as AdbClient
from multi_device_reader import MultiDeviceReader
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
        # self.input_controller = MultiDeviceReader()

    def get_dataset_len(self):
        x = len(os.listdir("generated/runs/dataset"))
        return x

    def start(self):
        if (self.current_run != None):
            return

        self.dataset = time.strftime(('%Y-%m-%d %H:%M:%S %Z'), time.localtime(time.time()))
        print(f"Starting Recording... Dataset: ${self.dataset}")
        self.save_que = SaveQue(self.dataset, f"generated/runs/dataset/{self.dataset}")
        self.current_run = InGameRun(self.controller, self.save_que)
        self.run_no += 1

    def stop(self):
        if (self.current_run == None): 
            return

        print(f"Stopping Recording...: {self.run_no}")
        self.current_run.close()
        self.current_run = None
        if (self.run_no % int((self.get_dataset_len() + 10) / 10) == 0):
            trainer.main()
            self.model, self.device = ssai_model.load("generated/models/test.pth")

    def tick(self):
        img_rgb = self.controller.capture()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self.rgb_queue.append(img_rgb)
        self.bgr_queue.append(img_bgr)
        gamestate = self.gsd.detect_gamestate(img_bgr)
        lane = self.gsd.detect_lane(img_bgr)

        if (self.current_run == None and gamestate == constants.GAME_STATE_OVER):
            self.controller.tap(400, 750)
            time.sleep(0.1)
            self.controller.tap(355, 101)

        if (self.current_run == None and gamestate < constants.GAME_STATE_OVER):
            self.start()

        if (self.current_run != None):
            if (len(self.rgb_queue) >= 3):
                action, logits = self.model.infer([Image.fromarray(x) for x in self.rgb_queue], self.current_run.run_secs(), self.device, randomize=self.get_dataset_len() <= 25)

                if (gamestate != constants.GAME_STATE_OVER):
                    self.current_run.give_command(action, list(self.bgr_queue), gamestate, logits, lane)

                self.current_run.tick(gamestate, lane)

            if (self.current_run.is_finished()):
                self.stop()
                self.current_run = None

        time.sleep(0.075)
        return True

    def start_mainloop(self):
        while True:
            try:
                if (not self.tick()):
                    break
            except KeyboardInterrupt as e:
                break
        self.stop()


def main():
    # model, device = None, None
    model, device = ssai_model.load("generated/models/test.pth") if os.path.exists("generated/models/test.pth") else (ssai_model.SSAIModel(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)

    logfile = open("generated/emu_log.txt", "w+")
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    emulator_utils.launch(adb_client, 15, stderror=logfile, stdout=logfile)
    player = Player(model, device)
    player.start_mainloop()
    logfile.close()
    

if __name__ == "__main__":
    main()