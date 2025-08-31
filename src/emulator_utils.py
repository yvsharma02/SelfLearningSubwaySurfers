import subprocess
from ppadb.client import Client
from ppadb.device import Device
import time

def launch(adb_client : Client, avd_name : str, timeout_s : float = 10.0, stdout = None, stderror = None):
    start_time = time.time()
    cmd = [
    "emulator",
    "-avd", avd_name,
    "-no-window",
    "-no-audio",
    "-no-boot-anim",
    "-grpc", "8554",
    "-idle-grpc-timeout", "0"
    ]

    process = subprocess.Popen(cmd, stdout=stderror, stderr=stdout)
    while len(adb_client.devices()) == 0:
        if ((time.time() - start_time) >= timeout_s):
            raise TimeoutError("Emulator Launch timed out")
        time.sleep(0.1)
    
    return adb_client.devices()[0]


def test_capture_rate(device : Device, run_duration_s = 10.0):
    start_time = time.time()
    capture_count = 0
    while ((time.time() - start_time) <= run_duration_s):
        ss = device.screencap()
        capture_count += 1

        print(f"{capture_count / (time.time() - start_time)} fps")