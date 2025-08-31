import selectors
import threading
import os

class PipeReader:
    READ_CHUNK_SIZE = 4096

    def __init__(self, src, selector : selectors.BaseSelector):
        self.src = src
        self.buffer = []
        self.running_data = ""
        self.lock = threading.Lock() # This should be redundant since selectors are single threaded.
        self.selector = selector

        selector.register(src, selectors.EVENT_READ, self.on_recieve)

    def on_recieve(self, fd):
        chunk_size = PipeReader.READ_CHUNK_SIZE
        data = os.read(fd, chunk_size)
        if (data):
            data_str = data.decode()
            with (self.lock):
                if (len(data_str) < chunk_size and not self.running_data):
                    self.buffer.append(data_str)
                elif (self.running_data and len(data_str) < chunk_size):
                    self.running_data += data_str
                    self.buffer.append(self.running_data)
                    self.running_data = ""
                elif (self.running_data and len(data_str) == chunk_size):
                    self.running_data += data_str
                else:
                    self.running_data = data_str
        else:
            self.selector.unregister(fd)

    def read_line(self):
        with (self.lock):
            if (len(self.buffer) == 0):
                return ""
            
            res = self.buffer[0]
            self.buffer = self.buffer[1:]
            return res

    def read_lines(self):
        with (self.lock):
            res = self.buffer
            self.buffer = []
            return res