# import threading
# import queue

# class AsyncWriter:

#     def __init__(self, filename, mode, max_time_bw_flush_sec = 5.0, max_req_before_flush = 25):
#         if (not "w" in mode and not "a" in mode):
#             raise Exception("Must be writable/appendable")
#         self.filename = filename
#         self.mode = mode
#         self.lock = threading.lock()
#         self.executor = threading.Thread()
#         self.write_queue = queue

#         self.write_buffer = []

#     def __enter__(self):
#         self.fd = open(self.filename, self.mode)

#     def __exit__(self):
#         self.fd.close()