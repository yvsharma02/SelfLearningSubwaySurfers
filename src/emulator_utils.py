import subprocess
import cv2
from ppadb.client import Client
from ppadb.device import Device
import time
import numpy as np
import matplotlib.pyplot as plt
import xvfb_capture

from grpc._channel import _InactiveRpcError
from grpc_controller import EmulatorController

def launch(adb_client : Client, timeout_s : float = 10.0, stdout = None, stderror = None, use_cpu_rendering = False):
    print("Opening Emulator")

    #FOR FUTURE ME: If I want to run this via X-forwarding, remove the -no-window option and remove the accel on and gpu command option.
    start_time = time.time()
    start_cmd = [
    "emulator",
    "-avd", "default_avd",
    # "-no-window",
    "-no-audio",
    "-no-boot-anim",
    "-grpc", "8554",
    "-idle-grpc-timeout", "0",
    "-no-snapshot-load"
    # "-gpu", ("swiftshader_indirect" if use_cpu_rendering else "host"),
    # "-accel", "on",
    ]

    subprocess.Popen(start_cmd, stdout=stdout, stderr=stderror)
    while len(adb_client.devices()) == 0:
        if ((time.time() - start_time) >= timeout_s):
            raise TimeoutError("Emulator Launch timed out")
        # print((time.time() - start_time) )
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
    
    
def test_capture_rate(run_duration_s = 60.0, capture_function = None):
    # Since we want to target 60 fps. We won't ever get 300, so I can safely leave it at 300.
    data_point_count = int(300 * run_duration_s)
    frame_start_times = [time.time()] * data_point_count
    capture_count = 0
#    frame_start_times[capture_count] = time.time()

    fsp_tracker = [0] * data_point_count
    fps_60_tracker = [0] * data_point_count

    while ((time.time() - frame_start_times[0]) <= run_duration_s):
        frame_start_times[capture_count] = time.time()
        res = capture_function()
        capture_count += 1

        fps = capture_count / (time.time() - frame_start_times[0])
        last_60_frames_fps = 0 if capture_count < 60 else 60 / (time.time() - frame_start_times[capture_count - 60])
        fsp_tracker[capture_count - 1] = fps
        fps_60_tracker[capture_count - 1] = last_60_frames_fps
        cv2.imwrite(f"generated/images/{capture_count}.png", res)
        # print(f"{fps}  ----  {last_60_frames_fps}")
    
    return fsp_tracker[max(min(capture_count - 1, 60), 60):capture_count], fps_60_tracker[max(min(capture_count - 1, 60) - 1, 60):capture_count]

def test_emulator(adb_client, cpu_rendering=False):
    with open(f"generated/emulator_log_{ 'cpu' if cpu_rendering else 'gpu'}.txt", "w+") as logfile:
        launch(adb_client, 10,  logfile, logfile, cpu_rendering)
        time.sleep(10)
        
        recorder = EmulatorController()
        done = False
        print("Waiting for emulator to start....")
        while (not done):
            try:
                fps, fps60 = test_capture_rate(15, recorder.capture)
                plt.figure(1)
                plt.title(f"{'CPU' if cpu_rendering else 'GPU'} FPS")
                plt.xlabel("frame #")
                plt.ylabel("fps")
                plt.plot([i for i in range(0, len(fps))], fps)

                plt.savefig(f"generated/{'cpu' if cpu_rendering else 'gpu'}_fps.png")

                plt.figure(2)
                plt.title(f"{'CPU' if cpu_rendering else 'GPU'} FPS-Last-60-Frames")
                plt.xlabel("frame #")
                plt.ylabel("fps")
                plt.plot([i for i in range(0, len(fps))], fps60)

                plt.savefig(f"generated/{'cpu' if cpu_rendering else 'gpu'}_fps-last-60.png")

                recorder.stop()
                kill(5, logfile, logfile)
                done = True
            except _InactiveRpcError as e:
                time.sleep(0.5)
                pass