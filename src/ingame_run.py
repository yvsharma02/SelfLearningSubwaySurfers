import time
import constants

class InGameRun:
    
    def __init__(self, gsd, emulator_controller, save_que):
        self.start_time = time.time()
        self.emulator_controller = emulator_controller
        self.save_que = save_que
        # self.gsd = gsd

        self.last_capture = None
        self.last_action = None,
        self.last_action_time = None
        self.last_action_state = None

    def run_secs(self):
        return time.time() - self.start_time

    def reaction_time(self):
        return 0.18 # Scale this with run_secs
    
    def next_action_delay(self):
        return 0.2 # Scale this with run_secs as well.

    def take_action(self, action, capture, gamestate):
        self.last_action_time = time.time()
        self.last_action = action
        self.last_capture = capture
        self.last_action_state = gamestate
        self.command_emulator(action)

    def time_since_last_action(self):
        return time.time() - (self.last_action_time if self.last_action_time is not None else self.start_time)

    def can_perform_action_now(self):
        return self.time_since_last_action() >= self.next_action_delay()

    def can_flush_last_action_now(self):
        return self.time_since_last_action() >= self.reaction_time()

    # Returns true if can be closed.
    def tick(self, new_state):
        if (self.can_flush_last_action_now()):
            self.flush(False)

        if (self.last_action_state != None):
            if (new_state > self.last_action_state):
                self.flush(True)

        return new_state == constants.GAME_STATE_OVER


    
    def flush(self, eliminate):
        if (eliminate):
            print("Eliminated!")
            self.save_que.put([i for i in range(0, 5) if i != self.last_action], self.last_capture, time.time() - self.start_time)

        self.last_action_time = None
        self.last_action_state = None
        self.last_capture = None

    def command_emulator(self, action):
        if (action is constants.ACTION_UP): self.emulator_controller.swipe_up()
        elif (action is constants.ACTION_DOWN): self.emulator_controller.swipe_down()
        elif (action is constants.ACTION_LEFT): self.emulator_controller.swipe_left()
        elif (action is constants.ACTION_RIGHT): self.emulator_controller.swipe_right()
    
    def close(self):
        self.save_que.close()