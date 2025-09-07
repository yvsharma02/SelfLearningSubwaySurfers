from inputs import devices
from threading import Thread, Lock

class DeviceReader:
    def __init__(self, device):
        self.device = device
        self.stop_signal = False
        self.lock = Lock()
        self.queue = []
        self.worker_thread = Thread(target=self.worker, daemon=True)
        self.worker_thread.start()

    def read(self):
        with (self.lock):
            res = self.queue
            self.queue = []
            return res

    def worker(self):
        while (not self.stop_signal):
            input_events = self.device.read() # Blocking
            with (self.lock):
                for input in input_events:
                    self.queue.append(input)


class MultiDeviceReader:
    def __init__(self):
        self.devices = [
            DeviceReader(x) for x in devices
        ]

    def read(self):
        res = []
        for device in self.devices:
            res = res + device.read()

        return res