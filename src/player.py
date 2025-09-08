from save_queue import SaveQue
from grpc_controller import EmulatorController
import actions
from ppadb.client import Client as AdbClient
from multi_device_reader import MultiDeviceReader
import time
import emulator_utils
import ssai_model
import gc

NOTHING_SAMPLING_RATE_ONE_IN_X = 30
# DEFAULT_KB_IDX = -1

keypress_action_map = {
    "NONE": actions.ACTION_NOTHING,
    "KEY_UP": actions.ACTION_UP,
    "KEY_DOWN": actions.ACTION_DOWN,
    "KEY_LEFT": actions.ACTION_LEFT,
    "KEY_RIGHT": actions.ACTION_RIGHT,
}

class Player:
    def __init__(self, controller, model=None, device=None):
        self.started = False
        self.controller = controller
        self.input_controller = MultiDeviceReader()
        self.model = model # If model is null, resort to manual play.
        self.device = device
        self.nothing_counter = 0
        self.auto_mode = model is None


    def start(self):
        if (self.started):
            return
        if (not self.auto_mode):
            self.dataset = time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(time.time()))
            print(f"Starting Recording... Dataset: ${self.dataset}\n")
            self.save_que = SaveQue(self.dataset, f"generated/runs/dataset/{self.dataset}")
            self.save_que.set_run_start_time()
            self.save_que.start()
        self.nothing_counter = 0
        self.started = True

    def stop(self):
        if (not self.started): 
            return
        if (not self.auto_mode):
            print("Stopping Recording...\n")
            self.started = False
            self.save_que.stop()

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
        
        if (self.model is None):
            self.manual_play(keypress)
        else:
            self.autoplay()
    

        time.sleep(0.01)
        return True

    def take_action(self, action):
        if (action is actions.ACTION_UP): self.controller.swipe_up()
        elif (action is actions.ACTION_DOWN): self.controller.swipe_down()
        elif (action is actions.ACTION_LEFT): self.controller.swipe_left()
        elif (action is actions.ACTION_RIGHT): self.controller.swipe_right()

        if (action == actions.ACTION_NOTHING):
            self.nothing_counter += 1

    def autoplay(self):
        if (not self.started):
            return
        img = self.controller.capture(False)
        confidence, action = self.model.infer(img, self.device)
        print(action, confidence)
        if (confidence > .6):
            self.take_action(action)

        del img
        gc.collect()
        time.sleep(0.1)

    def manual_play(self, keypress):
        if (not self.started):
            return
        if (keypress in keypress_action_map.keys()):
            action = keypress_action_map[keypress]
            
            if (action != actions.ACTION_NOTHING or self.nothing_counter % NOTHING_SAMPLING_RATE_ONE_IN_X == 0):
                self.save_que.put([x for x in range(0, len(keypress_action_map.keys())) if x is not action], self.controller.capture(True))
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
        # emulator_utils.kill()


def main():
    model = None
    model, device = ssai_model.load("generated/models/test.pth")

    logfile = open("generated/emu_log.txt", "w+")
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    emulator_utils.launch(adb_client, 15, stderror=logfile, stdout=logfile)
    player = Player(EmulatorController(), model=model, device=device)
    player.run()
    logfile.close()
    

main()