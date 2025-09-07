from inputs import get_key
from save_queue import SaveQue
from grpc_controller import EmulatorController
import actions
from ppadb.client import Client as AdbClient
import time
import emulator_utils

keypress_action_map = {
    "NONE": actions.ACTION_NOTHING,
    "KEY_UP": actions.ACTION_UP,
    "KEY_DOWN": actions.ACTION_DOWN,
    "KEY_LEFT": actions.ACTION_LEFT,
    "KEY_RIGHT": actions.ACTION_RIGHT,
}

class ManualPlayer:
    def __init__(self, controller):
        self.controller = controller
        dataset = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(time.time()))
        self.save_que = SaveQue(dataset, f"generated/runs/dataset/{dataset}")
        self.started = False

    # Called automatically.
    def start_internal(self):
        self.started = True
        self.save_que.set_run_start_time()
        self.save_que.start()

    def stop_internal(self):
        if (not self.started): 
            return
        
        self.started = False
        self.save_que.stop()

    def parse_key(self, event):
        if event.ev_type == "Key" and event.state == 1:
            if event.code == "KEY_UP": return "KEY_UP"
            elif event.code == "KEY_DOWN": return "KEY_DOWN"
            elif event.code == "KEY_LEFT": return "KEY_LEFT"
            elif event.code == "KEY_RIGHT": return "KEY_RIGHT"
            elif event.code == "KEY_Q": return "Q"
            elif event.code == "KEY_W": return "W"

        return None

    def update(self):
        events = get_key()
        keypress = "NONE"
        for event in events:
            keypress = self.parse_key(event)
        
            if (keypress == "Q"): 
                self.stop_internal()
                return False
            elif (keypress == "W"): 
                self.start_internal() 
                return True

        if (self.started and keypress in keypress_action_map.keys()):
            action = keypress_action_map[keypress]
            self.save_que.put([x for x in range(0, len(keypress_action_map.keys())) if x is not action], self.controller.capture())
            if (action is actions.ACTION_UP): self.controller.swipe_up()
            elif (action is actions.ACTION_DOWN): self.controller.swipe_down()
            elif (action is actions.ACTION_LEFT): self.controller.swipe_left()
            elif (action is actions.ACTION_RIGHT): self.controller.swipe_right()

        time.sleep(0.01)
        return True

    def run(self):
        print("W to start, Q to quit")
        while True:
            try:
                if (not self.update()):
                    break
            except KeyboardInterrupt as e:
                break
        self.stop_internal()
        emulator_utils.kill()


def main():
    logfile = open("generated/emu_log.txt", "w+")
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    emulator_utils.launch(adb_client, 15, stderror=logfile, stdout=logfile)
    # time.sleep(15)
    player = ManualPlayer(EmulatorController())
    player.run()
    logfile.close()

main()