from save_queue import SaveQue
from grpc_controller import EmulatorController
import custom_enums
from ppadb.client import Client as AdbClient
from multi_device_reader import MultiDeviceReader
import time
import emulator_utils
import ssai_model
import gc
import game_state_detector
from PIL import Image
import trainer

NOTHING_SAMPLING_RATE_ONE_IN_X = 30
RETRAIN_AFTER_X_RUNS = 25
# DEFAULT_KB_IDX = -1

keypress_action_map = {
    "NONE": custom_enums.ACTION_NOTHING,
    "KEY_UP": custom_enums.ACTION_UP,
    "KEY_DOWN": custom_enums.ACTION_DOWN,
    "KEY_LEFT": custom_enums.ACTION_LEFT,
    "KEY_RIGHT": custom_enums.ACTION_RIGHT,
}

class Player:
    def __init__(self, controller, model=None, device=None, record=False):
        self.gsd = game_state_detector.StateDetector()
        self.started = False
        self.controller = controller
        self.input_controller = MultiDeviceReader()
        self.model = model
        self.device = device
        self.nothing_counter = 0
        self.record = record
        self.auto_mode = model is not None
        self.last_state = None
        self.run_no = 0
        self.last_detected = False

    def start(self):
        if (self.started):
            return
        self.run_no += 1
        if (self.record):
            self.dataset = time.strftime(('%Y-%m-%d %H:%M:%S %Z' + ("-auto" if self.auto_mode else "")), time.localtime(time.time()))
            print(f"Starting Recording... Dataset: ${self.dataset}")
            self.save_que = SaveQue(self.dataset, f"generated/runs/dataset/{self.dataset}")
            self.save_que.set_run_start_time()
            self.save_que.start()
        self.nothing_counter = 0
        self.started = True

    def stop(self):
        if (not self.started): 
            return

        if (self.record):
            print("Stopping Recording...")
            self.started = False
            self.save_que.stop()

        if (self.run_no % RETRAIN_AFTER_X_RUNS == 0):
            trainer.main()

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
    

    def update(self):
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
            elif (keypress == "KEY_R"):
                self.stop()
                return True
            elif (keypress == "KEY_W"): 
                self.start() 
                return True
            
            break
        
        if (not self.auto_mode):
            self.manual_play(keypress)
        else:
            self.autoplay()
    

        time.sleep(0.01)
        return True

    def take_action(self, action):
        if (action is custom_enums.ACTION_UP): self.controller.swipe_up()
        elif (action is custom_enums.ACTION_DOWN): self.controller.swipe_down()
        elif (action is custom_enums.ACTION_LEFT): self.controller.swipe_left()
        elif (action is custom_enums.ACTION_RIGHT): self.controller.swipe_right()

        if (action == custom_enums.ACTION_NOTHING):
            self.nothing_counter += 1

        # print (f"Taking Action: {action}")

    def autoplay(self):
        img = self.controller.capture(True)
        state = self.gsd.detect_gamestate(img)

        if self.last_state == None:
            self.last_state = state
        else:
            self.last_state = state
            if (state == custom_enums.GAME_STATE_OVER):
                self.stop()
            else:
                self.start()

        if (state == custom_enums.GAME_STATE_ONGOING):
            nothing, confidence, action = self.model.infer(Image.fromarray(img), self.device)
            # print(f"Nothing confidence: {confidence}")
            if (confidence > .9):
                action = custom_enums.ACTION_NOTHING
            else:
                action += 1 # Reshift due to readdition of nothing.
            self.take_action(action)
#            self.save_ss(action, )

            # print(f"{action} : {confidence}")
            # if (confidence > .75):
            #     self.take_action(action)
            #     self.save_ss(action, img)
        else:
            self.controller.tap(400, 750)

        del img
        gc.collect()
        # time.sleep(0.1)

    def save_ss(self, action, capture):
        if (action != custom_enums.ACTION_NOTHING or self.nothing_counter % NOTHING_SAMPLING_RATE_ONE_IN_X == 0):
            self.save_que.put([x for x in range(0, len(keypress_action_map.keys())) if x is not action], capture)

    def manual_play(self, keypress):
        if (not self.started):
            return

        capture = self.controller.capture(True)
        det = self.gsd.detect_police(capture)
        if (self.last_detected is None or self.last_detected != det):
            self.last_detected = det
            print(f"Currently: {det}")
        self.save_ss(0, capture)   
        
        if (keypress in keypress_action_map.keys()):
            action = keypress_action_map[keypress]
            
            self.save_ss(action, capture)
            self.take_action(action)

    def run(self):
        print("W to start, R to reset, Q to quit\n")
        while True:
            try:
                if (not self.update()):
                    break
            except KeyboardInterrupt as e:
                break
            except Exception as e:
                print(e)
        self.stop()


def main():
    model, device = None, None
    # model, device = ssai_model.load("generated/models/test.pth")

    logfile = open("generated/emu_log.txt", "w+")
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    emulator_utils.launch(adb_client, 15, stderror=logfile, stdout=logfile)
    player = Player(EmulatorController(), model=model, device=device, record=True)
    player.run()
    logfile.close()
    

if __name__ == "__main__":
    main()