import subprocess
from ppadb.client import Client as AdbClient
import time
import sys
import threading
import selectors
import os

# Its 2025, why is it so painful to read output from a subprocess without blocking :#!
# Also why is there no way to flush buffer of a subprocess!!!? :<

class Device:
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
            chunk_size = Device.PipeReader.READ_CHUNK_SIZE
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

    def __init__(self, adb_client : AdbClient, avd_name = "headlessApi34", info_logfile = sys.stdout, error_logfile = sys.stderr):
        self.avd_name = avd_name
        self.adb_client = adb_client

        self.info_logfile = info_logfile
        self.error_logfile = error_logfile
        self.lock = threading.Lock()
        self.is_shutdown = False
        self.device = None

    def handle(self):
        self.handle_logs()

    def launch(self):
        print(f"Launching {self.avd_name}")
        self.process = subprocess.Popen(["emulator", "-avd", self.avd_name, "-no-window", "-no-audio", "-no-boot-anim"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        self.io_selector = selectors.DefaultSelector()
        self.info_reciever = Device.PipeReader(self.process.stdout.fileno(), self.io_selector)
        self.error_reciever = Device.PipeReader(self.process.stderr.fileno(), self.io_selector)
    
    def launched(self):
        return self.process != None

    def connect(self):
        print(f"Connecting to {self.avd_name}")
        self.device = self.adb_client.device(self.avd_name)
        print("Connected!")

    def connected(self):
        return self.device != None

    def handle_logs(self):
        events = self.io_selector.select(timeout=0)
        for key, mask in events:
            key.data(key.fileobj)

        for line in self.info_reciever.read_lines():
            self.on_info_recieve(line)

        for line in self.error_reciever.read_lines():
            self.on_error_recieve(line)


    def on_info_recieve(self, line : str):
        if (self.info_logfile != None):
            self.info_logfile.write(line)

        if ("Graphics Adapter Vendor Google" in line):
            self.connect()
            
    def on_error_recieve(self, line :str):
        if (self.error_logfile != None):
            self.error_logfile.write(line)

    def shutdown(self):
        if (not self.is_shutdown):
            print(f"Shutting down {self.avd_name}")

        try:
            command = "adb shell reboot -p"
            process = subprocess.run(command, shell=True, capture_output=True, text=True)

            if process.returncode == 0:
                print("Device shutdown command sent successfully.")
                self.is_shutdown = True
            else:
                print(f"Error sending shutdown command. Error: {process.stderr}")
        except FileNotFoundError:
            print("ADB command not found. Ensure ADB is installed and in your system's PATH.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def kill(self, try_shutdown = True):
        print(f"Killing {self.avd_name}")
        try:
            if (try_shutdown):
                self.shutdown()
        finally:
            self.process.kill()
            self.process.wait()

            self.clean()

    def clean(self):
        self.process = None
        self.device = None
        self.is_shutdown = False
        self.io_selector.close()

    def get_raw_device(self):
        if (self.device == None):
            raise Exception("Device not connected!")
        return self.device

# def launch_emulator():
#     time.sleep(5)
#     print(res)

# def connect_to_device():
#     client = AdbClient(host="127.0.0.1", port=5037)
#     device = client.device("headlessApi34")
#     return device

# def screen_capture(device):
#     result = device.screencap()
#     with open("generated/screen.png", "wb") as fp:
#         fp.write(result)