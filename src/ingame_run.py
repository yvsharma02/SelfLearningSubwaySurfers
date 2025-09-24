import time
import constants

class InGameRun:
    
    def __init__(self, gsd, emulator_controller, save_que):
        self.start_time = time.time()
        self.emulator_controller = emulator_controller
        self.save_que = save_que
        self.gsd = gsd

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

    def take_action(self, action, capture):
        self.last_action_time = time.time()
        self.last_action = action
        self.last_capture = capture
        self.last_action_state = self.gsd.detect_gamestate(capture)
        self.command_emulator(action)

    # Returns false when the game is over.
    def tick(self):
        if (self.last_action == None):
            return True
        
        action_time = time.time() - self.last_action_time()
        if (action_time < self.reaction_time()):
            return True
        
        new_state = self.gsd.detect_gamestate()

        if (new_state > self.last_action_state):
            self.save_que.put([i for i in range(0, 5) if i != self.last_action], self.last_capture, action_time)

        self.last_action_time = None
        self.last_action_state = None
        self.last_capture = None

        if (new_state == constants.GAME_STATE_OVER):
            return False
        
        return True

    def command_emulator(self, action):
        if (action is constants.ACTION_UP): self.controller.swipe_up()
        elif (action is constants.ACTION_DOWN): self.controller.swipe_down()
        elif (action is constants.ACTION_LEFT): self.controller.swipe_left()
        elif (action is constants.ACTION_RIGHT): self.controller.swipe_right()
        