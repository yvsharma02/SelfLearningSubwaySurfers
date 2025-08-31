import emulator_utils
import time
from ppadb.client import Client as AdbClient

def screen_capture(device):
    result = device.screencap()
    with open("generated/screen.png", "wb") as fp:
        fp.write(result)

def main():
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    log = open("generated/emulator_log.txt", "w+")

    print("Opening Emulator")
    device = emulator_utils.launch(adb_client, "headlessApi34", 10,  log, log)

if __name__ == "__main__":
    main()