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
        # self.started = False
        # self.controller = controller
        self.input_controller = MultiDeviceReader()
        # self.model = model
        # self.device = device
        # self.nothing_counter = 0
        # self.record = record
        # self.auto_mode = model is not None
        # self.last_state = None
        # self.run_no = 0
        # self.last_detected = False
        # self.current_run = None
        # # (Action, Action Time, Game State when Action was took)
        # self.last_action_data = (None, None, None)

    def start(self):
        if (self.current_run != None):
            return

        self.dataset = time.strftime(('%Y-%m-%d %H:%M:%S %Z' + ("-auto" if self.auto_mode else "")), time.localtime(time.time()))
        print(f"Starting Recording... Dataset: ${self.dataset}")
        self.save_que = SaveQue(self.dataset, f"generated/runs/dataset/{self.dataset}")
        self.current_run = InGameRun(self.gsd, self.controller, self.save_que)
        self.run_no += 1

    def stop(self):
        if (self.current_run == None): 
            return

        print("Stopping Recording...")
        self.current_run = None

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

        if (self.current_run == None):
            return True

        self.autoplay()
        return True

    def take_action(self, action):
        if (action is constants.ACTION_UP): self.controller.swipe_up()
        elif (action is constants.ACTION_DOWN): self.controller.swipe_down()
        elif (action is constants.ACTION_LEFT): self.controller.swipe_left()
        elif (action is constants.ACTION_RIGHT): self.controller.swipe_right()

        if (action == constants.ACTION_NOTHING):
            self.nothing_counter += 1

        # print (f"Taking Action: {action}")

    def autoplay(self):
        if(not self.current_run.tick()):
            self.current_run = None
            return
        
        if (self.current_run.last_action_time is not None or time.time() and self.current_run.last_action_time < self.current_run.next_action_delay()):
            return 

        img = self.controller.capture(True)
    
        action = self.model.infer(Image.fromarray(img), self.device)
        self.current_run.take_action(action, img)

        # img = self.controller.capture(True)

#         img = self.controller.capture(True)
#         state = self.gsd.detect_gamestate(img)

#         if self.last_state == None:
#             self.last_state = state
#         else:
#             self.last_state = state
#             if (state == constants.GAME_STATE_OVER):
#                 self.stop()
#             else:
#                 self.start()

#         if (state == constants.GAME_STATE_ONGOING):
#             nothing, confidence, action = self.model.infer(Image.fromarray(img), self.device)
#             # print(f"Nothing confidence: {confidence}")
#             if (confidence > .9):
#                 action = constants.ACTION_NOTHING
#             else:
#                 action += 1 # Reshift due to readdition of nothing.
#             self.take_action(action)
# #            self.save_ss(action, )

#             # print(f"{action} : {confidence}")
#             # if (confidence > .75):
#             #     self.take_action(action)
#             #     self.save_ss(action, img)
#         else:
#             self.controller.tap(400, 750)

#         del img
#         gc.collect()
        # time.sleep(0.1)

    # def save_ss(self, action, capture):
    #     if (action != constants.ACTION_NOTHING or self.nothing_counter % NOTHING_SAMPLING_RATE_ONE_IN_X == 0):
    #         self.save_que.put([x for x in range(0, len(keypress_action_map.keys())) if x is not action], capture)

    # def manual_play(self, keypress):
    #     if (not self.started):
    #         return

    #     capture = self.controller.capture(True)
    #     det = self.gsd.detect_police(capture)
    #     if (self.last_detected is None or self.last_detected != det):
    #         self.last_detected = det
    #         print(f"Currently: {det}")
    #     self.save_ss(0, capture)   
        
    #     if (keypress in keypress_action_map.keys()):
    #         action = keypress_action_map[keypress]
            
    #         self.save_ss(action, capture)
    #         self.take_action(action)

    def start_mainloop(self):
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
    player.start_mainloop()
    logfile.close()
    

if __name__ == "__main__":
    main()