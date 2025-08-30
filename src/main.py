import device as android_device
from ppadb.client import Client as AdbClient

if __name__ == "__main__":
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    log = open("generated/emulator_log.txt", "w+")
    device = android_device.Device(adb_client, info_logfile=log, error_logfile=log)
    device.launch()
    
    try:
        while (not device.connected()):
            if (len(adb_client.devices()) > 0):
                print(adb_client.devices()[0])
            device.handle()

        print("SSing")
        res = device.get_raw_device().screencap()
        with open("generated/screen.png", "wb") as fp:
            fp.write(res)
    except KeyboardInterrupt as e:
        print("\nClose Command Recieved.")
    finally:
        print("Closing Emulator Gracefully.")
        try:
            device.shutdown()
        finally:
            device.kill(try_shutdown=False)