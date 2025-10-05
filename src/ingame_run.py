import time
import constants
import cv2
import torch
import random
from collections import deque
from save_queue import SaveItem

def log(msg):
    print(msg)
    

class InGameRun:
    
    class Command:
        def __init__(self, capture, pred_action, state, elimination_window_min, elimination_window_max, logits, lane):
            self.capture = capture
            self.action = pred_action
            self.game_state = state
            self.command_time = time.time()
            self.elim_win_low = elimination_window_min
            self.elim_win_high = elimination_window_max
            self.execute_time = None
            self.logits = logits
            self.saved = False
            self.lane = lane

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

    def scale_time(self, *x):
        sf = 1 + self.run_secs() / (60 * 5)
        if (len(x) == 1):
            return x[0] / sf
        return tuple(v / sf for v in x)

    # Maybe cooldown should be independent frmo window_high???
    # Scale?? (1.5 times lower at 2 min mark???)
    def get_command_elim_window(self, action):
        def get_unscaled():
            if action == constants.ACTION_NOTHING:
                return 0, 0
            if action == constants.ACTION_UP:
                return 0.35, 0.7 #torch.normal(.8, .15, size=(1,)).item()# 1 + (random.random() - 0.5) * 2 * .35
            if action == constants.ACTION_DOWN:
                return 0.275, 0.65 #torch.normal(0.65, .050, size=(1,)).item()#0.55 + (random.random() - 0.5) * 2 * .05
            # point to note: left and right actions are mostly eliminated due to deflection or out of bounds.
            if action == constants.ACTION_LEFT:
                return 0.4, 0.65 #torch.normal(0.65, .0375, size=(1,)).item()#0.55 + (random.random() - 0.5) * 2 * .05
            if action == constants.ACTION_RIGHT:
                return 0.4, 0.675# torch.normal(0.65, .0375, size=(1,)).item()#0.55 + (random.random() - 0.5) * 2 * .05
            
        low, high = get_unscaled()
        return self.scale_time(low, high)
    
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

    def give_command(self, action, capture, gamestate, logits, lane):
        if (self.run_secs() < self.start_delay()):
            return

        now = time.time()

        if (action == constants.ACTION_NOTHING):
            self.nothing_buffer.append((action, capture, gamestate, logits, now, lane))
            return
        
        if (self.executing_cmd != None and self.executing_cmd.time_since_execution() < self.executing_cmd.elim_win_high):
            return
        
        win_low, win_high = self.get_command_elim_window(action)
        if (self.queued_cmd == None or self.queued_cmd.action != action):
            self.queued_cmd = InGameRun.Command(capture, action, gamestate, win_low, win_high, logits, lane)

        pass

    def flush_nothing_buffer(self, eliminate, criteria, record=True, debug_log="NA"):
        to_flush = [i for i in range(0, len(self.nothing_buffer)) if criteria(self.nothing_buffer[i])]
        if (record):
            for idx in to_flush:
                elim = [0] if eliminate else [i for i in range(1, 5)]
                self.record(self.nothing_buffer[idx][1], 0, eliminate, self.nothing_buffer[idx][4], self.nothing_buffer[idx][3], debug_log)
        self.nothing_buffer = [self.nothing_buffer[i] for i in range(0, len(self.nothing_buffer)) if i not in to_flush]

    # def record_stale_nothing(self, eliminate):
    #     now = time.time()
    #     self.record_nothing_buffer(eliminate, lambda x : (now - x[4]) >= 1)

    def eliminate_retroactively(self, criteria, debug_log_append):
        entries : list[SaveItem] = self.save_que.get_entries_filter(criteria)
        for entry in entries:
            entry.eliminated = True
            entry.debug_log += debug_log_append

    def tick(self, new_state, new_lane):
        if (self.run_secs() < self.start_delay() or self.finished):
            return
        
        if (self.executing_cmd == None and self.queued_cmd != None):
            self.execute_command(self.queued_cmd)
            self.queued_cmd = None

        if (self.executing_cmd != None):
            if (new_state != constants.GAME_STATE_OVER and ((self.executing_cmd.action == constants.ACTION_LEFT and self.executing_cmd.lane == constants.LEFT_LANE) or (self.executing_cmd.action == constants.ACTION_RIGHT and self.executing_cmd.lane == constants.RIGHT_LANE))):
                    log("None prev nothing eliminiated (out of bounds): " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                    self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="OUT_OF_BOUNDS_FLUSH")
                    self.record_cmd(self.executing_cmd, True, "OUT_OF_BOUNDS")
                    self.executing_cmd = None
            else:
                now = time.time()
                tse = self.executing_cmd.time_since_execution()
                self.flush_nothing_buffer(False, lambda x : (now - x[4]) >= self.scale_time(1) and x[4] <= self.executing_cmd.command_time, debug_log="COMMAND_FLUSH")
                if (tse >= self.executing_cmd.elim_win_high and new_state != constants.GAME_STATE_OVER):
                    if ((self.executing_cmd.action == constants.ACTION_LEFT or self.executing_cmd.action == constants.ACTION_RIGHT) and self.executing_cmd.lane == new_lane):
                        log("None prev nothing eliminiated (lane bounce): " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                        self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="LANE_BOUNCE_FLUSH")
                        self.record_cmd(self.executing_cmd, True, "LANE_BOUNCE")
                    else:
                        log("None prev nothing eliminiated (after window): " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                        self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="AFTER_WINDOW_FLUSH") # TODO: Make sure this eliminates only nothings that happened before the command executed.
                        self.record_cmd(self.executing_cmd, False, "AFTER_WINDOW")
                    self.executing_cmd = None
                elif (new_state == constants.GAME_STATE_OVER):
                    if (tse < self.executing_cmd.elim_win_low):
                        nothing_count = len([x for x in self.nothing_buffer if x[4] <= self.executing_cmd.command_time])
                        log("Last few nothing eliminiated): " + str(nothing_count))
                        self.flush_nothing_buffer(True, lambda x : (now - x[4]) <= self.scale_time(2 if x[0] == constants.ACTION_UP else 1.25) and (x[4] < self.executing_cmd.command_time), debug_log="BEFORE_WINDOW_FLUSH_ELIM") # Elimninate last few seconds of noting.
                        self.flush_nothing_buffer(False, lambda x : (now - x[4]) > self.scale_time(2 if x[0] == constants.ACTION_UP else 1.25) and (x[4] < self.executing_cmd.command_time), debug_log="BEFORE_WINDOW_FLUSH_NO_ELIM")
                        log("Eliminating last seconds of action retroactively")
                        self.eliminate_retroactively(lambda i, x: now - x.cmd_time <= self.scale_time(1.5 if x.action == constants.ACTION_UP else 1),"_RETRO_ELIM")
                        # self.record_cmd(self.executing_cmd, False, "BEFORE_WINDOW") #Just don't bother with this.
                        # Maybe retroactively eliminate previous action in this case?
                    elif (self.executing_cmd.elim_win_low <= tse and tse <= self.executing_cmd.elim_win_high):
                        log("None prev nothing eliminiated: " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                        self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="IN_WINDOW_FLUSH")
                        self.record_cmd(self.executing_cmd, True, "IN_WINDOW")
                    else:
                        log("None prev nothing eliminiated (Lost Condition): " + str(len([x for x in self.nothing_buffer if x[4] < self.executing_cmd.command_time])))
                        self.flush_nothing_buffer(False, lambda x : (x[4] < self.executing_cmd.command_time), debug_log="LOST_CONDITION_FLUSH")
                        self.record_cmd(self.executing_cmd, True, "LOST_CONDITION")         
                    self.executing_cmd = None
                else:
                    pass
                    # log(f"Game Ongoing: {new_state}; {tse}")
            
        else:
            log("No pending command")
            now = time.time()
            self.flush_nothing_buffer(False, lambda x : (now - x[4]) >= self.scale_time(1), debug_log="NO_COMMAND_FLUSH_NON_ELIM")
            if (new_state == constants.GAME_STATE_OVER):
                log ("Game Over without command")
                self.flush_nothing_buffer(True, lambda x : True, debug_log="NO_COMMAND_FLUSH_ELIM")


        if (new_state == constants.GAME_STATE_OVER):
            # self.record_nothing_buffer(True, lambda x : True)
            log("Finished")
            self.finished = True


    def is_finished(self):
        return self.finished
    
    def record_cmd(self, cmd : Command, eliminate, debug_log="NA"):
        if (cmd.saved): raise Exception("Command already saved")

        cmd.mark_as_saved()

        capture = cmd.capture
        logits = cmd.logits
        ts = cmd.command_time

        self.record(capture, cmd.action, eliminate, ts, logits, debug_log)

    def record(self, capture, action, elim, cmd_time, logits, debug_log="NA"):
        self.save_que.put(action, elim, capture, cmd_time, logits, debug_log)

    def execute_command(self, cmd : Command):
        if (cmd.time_since_execution() >= 0): raise Exception("Command Already Executed.")

        action = cmd.action
        if action == constants.ACTION_UP: self.emulator_controller.swipe_up()
        elif action == constants.ACTION_DOWN: self.emulator_controller.swipe_down()
        elif action == constants.ACTION_LEFT: self.emulator_controller.swipe_left()
        elif action == constants.ACTION_RIGHT: self.emulator_controller.swipe_right()
        cmd.mark_as_executed()
        self.executing_cmd = cmd
    
    def close(self):
        self.save_que.close()