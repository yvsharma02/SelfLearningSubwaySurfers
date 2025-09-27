from save_queue import SaveQue
from grpc_controller import EmulatorController
import constants
from ppadb.client import Client as AdbClient
from multi_device_reader import MultiDeviceReader
import time
import emulator_utils
import ssai_model
import gc
import game_state_detector
from PIL import Image
import trainer
import torch
from ingame_run import InGameRun


NOTHING_SAMPLING_RATE_ONE_IN_X = 30
RETRAIN_AFTER_X_RUNS = 25
# DEFAULT_KB_IDX = -1

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
        self.input_controller = MultiDeviceReader()

    def start(self):
        if (self.current_run != None):
            return

        self.dataset = time.strftime(('%Y-%m-%d %H:%M:%S %Z'), time.localtime(time.time()))
        print(f"Starting Recording... Dataset: ${self.dataset}")
        self.save_que = SaveQue(self.dataset, f"generated/runs/dataset/{self.dataset}")
        self.current_run = InGameRun(self.gsd, self.controller, self.save_que)
        self.run_no += 1

    def stop(self):
        if (self.current_run == None): 
            return

        print(f"Stopping Recording...: {self.run_no}")
        self.current_run.close()
        self.current_run = None

        if (self.run_no % RETRAIN_AFTER_X_RUNS == 0):
            trainer.main()
            self.model, self.device = ssai_model.load("generated/models/test.pth")

    def is_valid_kb_down_event(self, event):
        if event.ev_type == "Key" and event.state == 1:
            if event.code == "KEY_UP": return True
            elif event.code == "KEY_DOWN": return True
            elif event.code == "KEY_LEFT": return True
            elif event.code == "KEY_RIGHT": return True
            elif event.code == "KEY_Q": return True
            elif event.code == "KEY_W": return True
            elif event.code == "KEY_R": return True

        return False
    

    def tick(self):
        keypress = "NONE"

        events = self.input_controller.read()
        for event in events:
            is_valid = self.is_valid_kb_down_event(event)
            if (not is_valid):
                continue
            
            keypress = event.code

            if (keypress == "KEY_Q"): 
                self.stop()
                return False

            break
        
        img = self.controller.capture(True)
        gamestate = self.gsd.detect_gamestate(img)

        if (self.current_run != None and gamestate == constants.GAME_STATE_OVER):
            self.stop()

        if (self.current_run == None):
            if (gamestate == constants.GAME_STATE_OVER):
                self.controller.tap(400, 750)
            else:
                self.start()

        if (self.current_run != None):
            self.autoplay(img, gamestate)
        
        time.sleep(0.05)
        return True

    def autoplay(self, img, gamestate):
        self.current_run.tick(gamestate)
        if (self.current_run.can_perform_action_now()):
            action, confidence = self.model.infer(Image.fromarray(img), self.current_run.run_secs(), self.device)
            self.current_run.take_action(action, img, gamestate)


    def start_mainloop(self):
        print("Press Q to quit\n")
        while True:
            try:
                if (not self.tick()):
                    break
            except KeyboardInterrupt as e:
                break
            except Exception as e:
                print(e)
        self.stop()


def main():
    model, device = None, None
    # model, device = ssai_model.load("generated/models/test.pth")
    model, device = ssai_model.SSAIModel(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logfile = open("generated/emu_log.txt", "w+")
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    emulator_utils.launch(adb_client, 15, stderror=logfile, stdout=logfile)
    player = Player(model, device)
    player.start_mainloop()
    logfile.close()
    

if __name__ == "__main__":
    main()