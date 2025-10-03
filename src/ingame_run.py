import time
import constants
import cv2
import torch
import random
from collections import deque


def log(msg):
    logfile = open("global_log.txt", "a+")
    logfile.write(f"{msg}\n")
    print(msg)
    

class InGameRun:
    
    class Command:
        def __init__(self, capture, pred_action, state, elimination_window_min, elimination_window_max, logits):
            self.capture = capture
            self.action = pred_action
            self.game_state = state
            self.command_time = time.time()
            self.elim_win_low = elimination_window_min
            self.elim_win_high = elimination_window_max
            self.execute_time = None
            self.logits = logits
            self.saved = False

        def time_since_given(self):
            return time.time() - self.command_time
        
        def time_since_execution(self):
            if self.execute_time is None:
                return -1
            return time.time() - self.execute_time
        
        def mark_as_executed(self):
            self.execute_time = time.time()

        def is_saved(self):
            return self.saved
        
        def mark_as_saved(self):
            self.saved = True

        def is_complete(self):
            return self.saved and self.time_since_execution() > self.elim_win_high

    def get_command_elim_window(self, action):
        if action == constants.ACTION_NOTHING:
            return 0, 0
        if action == constants.ACTION_UP:
            return 0.1, 0.8 + (random.random() - 0.5) * 2 * .45
        if action == constants.ACTION_DOWN:
            return 0.1, 0.45 + (random.random() - 0.5) * 2 * .125
        if action == constants.ACTION_LEFT:
            return 0.1, 0.45 + (random.random() - 0.5) * 2 * .125
        if action == constants.ACTION_RIGHT:
            return 0.1, 0.45 + (random.random() - 0.5) * 2 * .125
    
    def __init__(self, emulator_controller, save_que):
        self.start_time = time.time()
        self.emulator_controller = emulator_controller
        self.save_que = save_que
        self.nothing_buffer = []
        self.executing_cmd = None
        self.queued_cmd = None
        self.finished = False

    def run_secs(self):
        return time.time() - self.start_time

    def start_delay(self):
        return 2

    def give_command(self, action, capture, gamestate, logits):
        if (self.run_secs() < self.start_delay()):
            return

        now = time.time()

        if (action == constants.ACTION_NOTHING):
            self.nothing_buffer.append((action, capture, gamestate, logits, now))
            return
        
        if (self.executing_cmd != None and self.executing_cmd.time_since_execution() < self.executing_cmd.elim_win_high):
            return
        
        win_low, win_high = self.get_command_elim_window(action)
        if (self.queued_cmd == None or self.queued_cmd.action != action):
            self.queued_cmd = InGameRun.Command(capture, action, gamestate, win_low, win_high, logits)

        pass

    def flush_nothing_buffer(self, eliminate, criteria, record=True, debug_log="NA"):
        to_flush = [i for i in range(0, len(self.nothing_buffer)) if criteria(self.nothing_buffer[i])]
        if (record):
            for idx in to_flush:
                elim = [0] if eliminate else [i for i in range(1, 5)]
                self.record(elim, self.nothing_buffer[idx][1], self.nothing_buffer[idx][4], self.nothing_buffer[idx][3], debug_log)
        self.nothing_buffer = [self.nothing_buffer[i] for i in range(0, len(self.nothing_buffer)) if i not in to_flush]

    # def record_stale_nothing(self, eliminate):
    #     now = time.time()
    #     self.record_nothing_buffer(eliminate, lambda x : (now - x[4]) >= 1)

    def tick(self, new_state):
        if (self.run_secs() < self.start_delay() or self.finished):
            return
        
        if (self.executing_cmd == None and self.queued_cmd != None):
            self.execute_command(self.queued_cmd)
            self.queued_cmd = None

        if (self.executing_cmd != None):
            now = time.time()
            self.flush_nothing_buffer(False, lambda x : (now - x[4]) >= 1 and x[4] <= self.executing_cmd.command_time, debug_log="COMMAND_FLUSH")
            if (self.executing_cmd.time_since_execution() >= self.executing_cmd.elim_win_high and new_state != constants.GAME_STATE_OVER):
                log("None prev nothing eliminiated (after window): " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="AFTER_WINDOW_FLUSH") # TODO: Make sure this eliminates only nothings that happened before the command executed.
                self.record_cmd(self.executing_cmd, False, "AFTER_WINDOW")
                self.executing_cmd = None
            elif (new_state == constants.GAME_STATE_OVER):
                if (self.executing_cmd.time_since_execution() < self.executing_cmd.elim_win_low):
                    log("All prev nothing eliminiated: " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                    self.flush_nothing_buffer(True, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="BEFORE_WINDOW_FLUSH") # Elimninate last few seconds of noting.
                    # self.record_cmd(self.executing_cmd, False, "BEFORE_WINDOW") #Just don't bother with this.
                elif (self.executing_cmd.elim_win_low <= self.executing_cmd.time_since_execution() and self.executing_cmd.time_since_execution() <= self.executing_cmd.elim_win_high):
                    log("None prev nothing eliminiated: " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                    self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="IN_WINDOW_FLUSH")
                    self.record_cmd(self.executing_cmd, True, "IN_WINDOW")
                else:
                    log("All prev nothing eliminiated (Lost Condition): " + str(len([x for x in self.nothing_buffer if x[4] >= self.executing_cmd.command_time + self.executing_cmd.elim_win_high])))
                    self.flush_nothing_buffer(True, lambda x : (x[4] >= self.executing_cmd.command_time + self.executing_cmd.elim_win_high), debug_log="LOST_CONDITION")            
                self.executing_cmd = None
            else:
                log("???????????????????")
            
        else:
            log("No pending command")
            now = time.time()
            self.flush_nothing_buffer(False, lambda x : (now - x[4]) >= 1, debug_log="NO_COMMAND_FLUSH_NON_ELIM")
            if (new_state == constants.GAME_STATE_OVER):
                self.flush_nothing_buffer(True, lambda x : True, debug_log="NO_COMMAND_FLUSH_ELIM")


        if (new_state == constants.GAME_STATE_OVER):
            # self.record_nothing_buffer(True, lambda x : True)
            self.finished = True


    def is_finished(self):
        return self.finished
    
    def record_cmd(self, cmd : Command, eliminate, debug_log="NA"):
        if (cmd.saved): raise "Command already saved"

        cmd.mark_as_saved()

        capture = cmd.capture
        logits = cmd.logits
        ts = cmd.command_time
        elim = [cmd.action] if eliminate else [i for i in range(0, 5) if i != cmd.action]

        self.record(elim, capture, ts, logits, debug_log)

    def record(self, eliminations, capture, cmd_time, logits, debug_log="NA"):
        _, act_max_idx = torch.max(logits, dim=0)
        _, act_min_idx = torch.min(logits, dim=0)
        act_max = constants.action_to_name(act_max_idx.item())
        act_min = constants.action_to_name(act_min_idx.item())
        log("Eliminated: [" + ",".join([constants.action_to_name(x) for x in eliminations]) + "] ; Logits: [" + ", ".join([f'{x:.4f}' for x in logits]) + "]; " + "ELIM_MAX: " + act_max + "; ELIM_MIN: " + act_min + "; DEBUG: " + debug_log)
        self.save_que.put(eliminations, capture, cmd_time, logits, debug_log)

    def execute_command(self, cmd : Command):
        if (cmd.time_since_execution() >= 0): raise "Command Already Executed."

        action = cmd.action
        if action == constants.ACTION_UP: self.emulator_controller.swipe_up()
        elif action == constants.ACTION_DOWN: self.emulator_controller.swipe_down()
        elif action == constants.ACTION_LEFT: self.emulator_controller.swipe_left()
        elif action == constants.ACTION_RIGHT: self.emulator_controller.swipe_right()
        cmd.mark_as_executed()
        self.executing_cmd = cmd
    
    def close(self):
        self.save_que.close()
