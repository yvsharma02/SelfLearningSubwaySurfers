from queue import Queue
from threading import Thread
import os
import cv2
import constants
import torch

class SaveItem:
    def __init__(self, im_no, action, eliminated, time_sec, img, logits, debug_log):
        self.action = action
        self.eliminated = eliminated
        self.im_no = im_no
        # self.elimiated_choices = eliminated_choices
        self.time_sec = time_sec
        self.img = img
        self.logits = logits
        self.debug_log = debug_log

class SaveQue:
    def __init__(self, dataset_name, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.counter = 0
        self.remaining_metadata = Queue()
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.metadata_file = open(os.path.join(self.dataset_dir, "metadata.txt"), "w+")

    def put(self, action, eliminated, img, time_s, logits, debug_log):
        item = SaveItem(self.counter, action, eliminated, time_s, img, logits, debug_log)
        self.process_image(item)
        self.log(item)
        self.counter += 1

    def log(self, item):
        _, act_max_idx = torch.max(item.logits, dim=0)
        _, act_min_idx = torch.min(item.logits, dim=0)
        act_max = constants.action_to_name(act_max_idx.item())
        act_min = constants.action_to_name(act_min_idx.item())
        eliminations = self.get_elim_list(item.action, item.eliminated)
        print("Eliminated: [" + ",".join([constants.action_to_name(x) for x in eliminations]) + "] ; Logits: [" + ", ".join([f'{x:.4f}' for x in item.logits]) + "]; " + "ELIM_MAX: " + act_max + "; ELIM_MIN: " + act_min + "; DEBUG: " + item.debug_log)

    def get_elim_list(self, action, elim):
        return [action] if elim else [i for i in range(0, 5) if i != action]

    def process_metadata_entries(self):
        for item in self.remaining_metadata:
            eliminated_choices = self.get_elim_list(item.action, item.eliminated)
            eliminated_actions_str = f"[{','.join([str(act) for act in eliminated_choices])}]"
            logits_str = f"({",".join(str(l.item()) for l in item.logits)})"
            self.metadata_file.write(f"{item.im_no}; {item.time_sec}; {eliminated_actions_str}; {logits_str}; {item.debug_log}\n")

    def process_image(self, item):
        cv2.imwrite(os.path.join(self.dataset_dir, f"{item.im_no}.png"), item.img)
        del item.img
        item.img = None
        self.remaining_metadata.put(item)
        
        return True
    
    def count(self):
        return len(self.remaining_metadata)
    
    def get_entry(self, idx):
        return self.remaining_metadata[idx]
    
    def get_entries(self, indices):
        return [self.remaining_metadata[x] for x in indices]
    
    def get_entries_filter(self, filter):
        return [x for x in self.remaining_metadata if filter(x)]

    def close(self):
        self.process_metadata_entries()
        self.metadata_file.close()