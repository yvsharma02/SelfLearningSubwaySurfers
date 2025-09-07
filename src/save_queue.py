from queue import Queue
from threading import Thread
import time
import os
import cv2
import gc

class SaveItem:
    def __init__(self, im_no, eliminated_choices, time_sec, img):
        self.im_no = im_no
        self.elimiated_choices = eliminated_choices
        self.time_sec = time_sec
        self.img = img

class SaveQue:
    def __init__(self, dataset_name, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
#        self.queue = Queue()
        self.running = False

    # True start
    def set_run_start_time(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.run_start_time = time.time()

    def put(self, eliminated_choices, img):
        item = SaveItem(self.counter, eliminated_choices, (time.time() - self.run_start_time), img)
#        self.queue.put(item)
        self.process(item)
        self.counter += 1

    def stop(self):
        self.stop_signal_given = True
#        self.put(None, None) # Dummy to close queue.

    def stop_internal(self):
        self.run_start_time = None
        self.counter = -1
        self.running = False
        self.metadata_file.close()
    
    def start(self):
        self.counter = 0
        self.running = True
        self.stop_signal_given = False
#        self.worker_thread = Thread(target=self.worker)
#        self.worker_thread.start()
        self.metadata_file = open(os.path.join(self.dataset_dir, "metadata.txt"), "w+")

    # Return false means end main loop
    def process(self, item):
        if (item.elimiated_choices is None and item.img is None):
            return False

        cv2.imwrite(os.path.join(self.dataset_dir, f"{item.im_no}.png"), item.img)
        eliminated_actions_str = f"[{','.join([str(act) for act in item.elimiated_choices])}]"
        self.metadata_file.write(f"{item.im_no}; {item.time_sec}; {eliminated_actions_str}\n")
        print(f"Frame: ${item.im_no}")
        del item
        gc.collect()
        return True

    def worker(self):
        # Technically even im no can be derived from number number.
        while (not self.stop_signal_given):
            item : SaveItem = self.queue.get()

            if (not self.process(item)):
                break

        self.stop_internal()