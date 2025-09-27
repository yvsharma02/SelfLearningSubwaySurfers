import time
import constants
import cv2
import torch

class InGameRun:
    
    ACTIONS_WAIT_TIME = [
        0.1, 0.75, 0.5, 0.5, 0.5
        # 0.075,
        # 0.4125,
        # 0.35,
        # 0.3,
        # 0.3
    ]

    def wait_time_for_action(action):
        if (action == constants.ACTION_NOTHING):
            return torch.normal(0.15, 0.05, size=(1,)).item()
        
        if (action == constants.ACTION_UP):
            return torch.normal(.75, .33, size=(1,)).item()
        
        if (action == constants.ACTION_DOWN):
            return torch.normal(.55, .1, size=(1,)).item()
        
        if (action == constants.ACTION_LEFT):
            return torch.normal(.55, .1, size=(1,)).item()
        
        if (action == constants.ACTION_RIGHT):
            return torch.normal(.55, .1, size=(1,)).item()


    def __init__(self, gsd, emulator_controller, save_que):
        self.start_time = time.time()
        self.emulator_controller = emulator_controller
        self.save_que = save_que
        # self.first_normal_state_detected = False

        self.last_capture = None
        self.last_action = None
        self.last_action_time = None
        self.last_action_state = None
        self.last_logits = None

        self.first_tick = False

    def run_secs(self):
        return time.time() - self.start_time


    def reaction_time(self):
        if (self.last_action == None):
            return 0
        else:
            return InGameRun.wait_time_for_action(self.last_action)
    
    # def next_action_delay(self):
        # return 0.4 # Scale this with run_secs as well.

    def start_delay(self):
        return 2

    def take_action(self, action, capture, gamestate, logits):
        self.last_action_time = time.time()
        self.last_action = action
        self.last_capture = capture
        self.last_action_state = gamestate
        self.last_logits = logits
        # print(f"Taking Action: {action}")
        self.command_emulator(action)

    def time_since_last_action(self):
        return time.time() - (self.last_action_time if self.last_action_time is not None else self.start_time)

    def can_perform_action_now(self):
        return self.last_action is None and self.run_secs() >= self.start_delay()
        # return self.time_since_last_action() >= self.next_action_delay()

    def can_flush_last_action_now(self):
        return self.time_since_last_action() >= self.reaction_time()

    def tick(self, new_state):
        if (self.run_secs() < self.start_delay()):
            return
        
        if (not self.first_tick):
            self.first_tick = True
            print("Truly Started!")

        
        # if (not self.first_normal_state_detected):
        #     if (new_state == constants.GAME_STATE_NON_FATAL_MISTAKE):
        #         return
        #     print(f"First Normal: ", float(self.run_secs() ))
        #     self.first_normal_state_detected = True

        if (self.last_action_state != None):
            if (new_state > self.last_action_state):
                self.flush(True)

        if (self.can_flush_last_action_now()):
            self.flush(False)


    def flush(self, eliminate):
        if (self.last_action != None):
            _, act_max_idx = torch.max(self.last_logits, dim=0)
            _, act_min_idx = torch.min(self.last_logits, dim=0)
            act_max = constants.action_to_name(act_max_idx.item())
            act_min = constants.action_to_name(act_min_idx.item())
            print(("Eliminated: " if eliminate else "Validated: ") + constants.action_to_name(self.last_action) + " ; Logits: [" + ", ".join([f'{x:.4f}' for x in self.last_logits]) + "]; " + "ELIM_MAX: " + act_max + "; ELIM_MIN: " + act_min)
            if (eliminate):
                # print(f"Eliminated!: {self.last_action}")
                self.save_que.put([self.last_action], self.last_capture, self.run_secs(), self.last_logits)
            else:
                # print(f"Did Not Eliminate!: {self.last_action}")
                self.save_que.put([i for i in range(0, 5) if i != self.last_action], self.last_capture, self.run_secs(), self.last_logits)

        self.last_action = None
        self.last_action_state = None
        self.last_capture = None
        self.last_logits = None

    def command_emulator(self, action):
        if (action == constants.ACTION_UP): self.emulator_controller.swipe_up()
        elif (action == constants.ACTION_DOWN): self.emulator_controller.swipe_down()
        elif (action == constants.ACTION_LEFT): self.emulator_controller.swipe_left()
        elif (action == constants.ACTION_RIGHT): self.emulator_controller.swipe_right()
    
    def close(self):
        if ((self.last_action is not None) and (self.last_capture is not None)):
            self.flush(True)
        self.save_que.close()