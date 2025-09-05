import emulator_utils
import time

from grpc_capture import Recorder
from ppadb.client import Client as AdbClient

def main():
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    emulator_utils.test_emulator(adb_client, True)
    # emulator_utils.test_emulator(adb_client, False)

if __name__ == "__main__":
    main()