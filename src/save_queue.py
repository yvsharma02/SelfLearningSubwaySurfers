from queue import Queue
from threading import Thread
import os
import cv2
import constants
import torch

class SaveItem:
    def __init__(self, im_no, action, eliminated, cmd_time, img, logits, debug_log, frame_number):
        self.action = action
        self.eliminated = eliminated
        self.im_no = im_no
        # self.elimiated_choices = eliminated_choices
        self.cmd_time = cmd_time
        self.img = img
        self.logits = logits
        self.debug_log = debug_log
        self.frame_number = frame_number

class SaveQue:
    def __init__(self, dataset_name, dataset_dir):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.counter = 0
        self.remaining_metadata = []
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.metadata_file = open(os.path.join(self.dataset_dir, "metadata.txt"), "w+")

    def put(self, action, eliminated, img, time_s, logits, frame_number, debug_log):
        item = SaveItem(self.counter, action, eliminated, time_s, img, logits, debug_log, frame_number)
        self.process_image(item)
        self.log(item)
        self.counter += 1

    def log(self, item):
        _, act_max_idx = torch.max(item.logits, dim=0)
        _, act_min_idx = torch.min(item.logits, dim=0)
        act_max = constants.action_to_name(act_max_idx.item())
        act_min = constants.action_to_name(act_min_idx.item())
        eliminations = self.get_elim_list(item.action, item.eliminated)
        print("Eliminated: [" + ",".join([constants.action_to_name(x) for x in eliminations]) + "] ; Logits: [" + ", ".join([f'{x:.4f}' for x in item.logits]) + "]; " + "ELIM_MAX: " + act_max + "; ELIM_MIN: " + act_min + "; DEBUG: " + item.debug_log + "; FRAME_NUMBER: " + str(item.frame_number))

    def get_elim_list(self, action, elim):
        return [action] if elim else [i for i in range(0, 5) if i != action]

    def process_metadata_entries(self):
        for item in self.remaining_metadata:
            eliminated_choices = self.get_elim_list(item.action, item.eliminated)
            eliminated_actions_str = f"[{','.join([str(act) for act in eliminated_choices])}]"
            logits_str = f"({",".join(str(l.item()) for l in item.logits)})"
            self.metadata_file.write(f"{item.im_no}; {item.cmd_time}; {eliminated_actions_str}; {logits_str}; {item.debug_log}; {item.frame_number}\n")

    def process_image(self, item):
        for i in range(0, len(item.img)):
            cv2.imwrite(os.path.join(self.dataset_dir, f"{item.im_no}_{i}.png"), item.img[i])
        del item.img
        item.img = None
        self.remaining_metadata.append(item)
        
        return True
    
    def count(self):
        return len(self.remaining_metadata)
    
    def get_entry(self, idx):
        return self.remaining_metadata[idx]
    
    def get_entries(self, indices):
        return [self.remaining_metadata[x] for x in indices]
    
    #Filter peekback idx, item
    def get_entries_filter(self, filter):
        return [x for i, x in enumerate(self.remaining_metadata) if filter(len(self.remaining_metadata) - 1 - i, x)]

    def close(self):
        self.process_metadata_entries()
        self.metadata_file.close()