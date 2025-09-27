from queue import Queue
from threading import Thread
import os
import cv2
import gc

class SaveItem:
    def __init__(self, im_no, eliminated_choices, time_sec, img, logits):
        self.im_no = im_no
        self.elimiated_choices = eliminated_choices
        self.time_sec = time_sec
        self.img = img
        self.logits = logits

class SaveQue:
    def __init__(self, dataset_name, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.counter = 0
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.metadata_file = open(os.path.join(self.dataset_dir, "metadata.txt"), "w+")

    def put(self, eliminated_choices, img, time_s, logits):
        item = SaveItem(self.counter, eliminated_choices, time_s, img, logits)
        self.process(item)
        self.counter += 1

    def process(self, item):
        cv2.imwrite(os.path.join(self.dataset_dir, f"{item.im_no}.png"), item.img)
        eliminated_actions_str = f"[{','.join([str(act) for act in item.elimiated_choices])}]"
        logits_str = f"({",".join(str(l.item()) for l in item.logits)})"
        self.metadata_file.write(f"{item.im_no}; {item.time_sec}; {eliminated_actions_str}; {logits_str}\n")
        del item
        gc.collect()
        return True
    
    def close(self):
        self.metadata_file.close()