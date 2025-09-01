import subprocess
from ppadb.client import Client
from ppadb.device import Device
import time

def launch(adb_client : Client, timeout_s : float = 10.0, stdout = None, stderror = None):
    print("Opening Emulator")

    start_time = time.time()
    start_cmd = [
    "emulator",
    "-avd", "default_avd",
    "-no-window",
    "-no-audio",
    "-no-boot-anim",
    "-grpc", "8554",
    "-idle-grpc-timeout", "0",
    "-gpu", "swiftshader_indirect",
    # "-accel", "on"
    ]

    subprocess.Popen(start_cmd, stdout=stdout, stderr=stderror)
    print("A")
    while len(adb_client.devices()) == 0:
        print("B")
        if ((time.time() - start_time) >= timeout_s):
            print("C")
            raise TimeoutError("Emulator Launch timed out")
        print((time.time() - start_time) )
        time.sleep(0.1)

    print("Emulator Launched")
    
    
    return adb_client.devices()[0]

def kill(timeout_s = 5, stdout = None, stderror = None):
    start_time = time.time()
    cmd_shutdown = ["adb", "-s", "emulator-5554", "emu", "kill"]
    cmd_force_kill = ["pkill", "-f", "qemu-system"]

    print("Shutting down")
    subprocess.Popen(cmd_shutdown, stdout=stdout, stderr=stderror)
    while (time.time() - start_time) < timeout_s:
        time.sleep(0.1)
    print("Killing")
    subprocess.Popen(cmd_force_kill, stdout=stdout, stderr=stderror)
    
    
def test_capture_rate(run_duration_s = 10.0, capture_function = None, setup_capture_function = None):
    start_time = time.time()
    capture_count = 0

    if (setup_capture_function != None):
        setup_capture_function()

    while ((time.time() - start_time) <= run_duration_s):
        capture_function()
        capture_count += 1

        print(f"{capture_count / (time.time() - start_time)} fps")