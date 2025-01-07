import time
from datetime import datetime

class Player:
    def __init__(self, visualizer_socket, playout_buffer, target_fps):
        self.visualizer_socket = visualizer_socket
        self.playout_buffer = playout_buffer
        self.target_fps = target_fps

    def send_to_visualizer(self):
        while True:
            timestamp = datetime.now().timestamp()

            if self.playout_buffer.qsize() > 0:
                data = self.playout_buffer.get()
                self.visualizer_socket.send(data)

            sleep_time = max(0, (1 / self.target_fps) - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)
