import emulator_utils
import time
from grpc._channel import _InactiveRpcError
from grpc_capture import Recorder
from ppadb.client import Client as AdbClient

def main():
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    with open("generated/emulator_log.txt", "w+") as logfile:
        device = emulator_utils.launch(adb_client, 15,  logfile, logfile)
        
        recorder = Recorder()
        done = False
        print("Waiting for emulator to start....")
        while (not done):
            try:
                emulator_utils.test_capture_rate(5, recorder.capture, recorder.setup)
                recorder.stop()
                emulator_utils.kill(5, logfile, logfile)
                done = True
            except _InactiveRpcError as e:
                time.sleep(0.5)
                pass

if __name__ == "__main__":
    main()