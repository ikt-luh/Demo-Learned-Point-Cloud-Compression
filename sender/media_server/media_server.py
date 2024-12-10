import time
import os
import zmq
import pickle
import threading
from server import HTTPServerHandler
from mpd_manager import MPDManager

class StreamingServer:
    def __init__(self, output_directory="./media", zmq_address="tcp://*:5556", port=8080):
        # Setup directories and MPD Manager
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        print(self.output_directory)

        self.mpd_manager = MPDManager(self.output_directory)
        self.mpd_manager.setup_adaptation_set()

        # Setup ZeroMQ
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(zmq_address)

        self.port = port
        self.segment_index = 1

    def start_http_server(self):
        HTTPServerHandler.start(directory=self.output_directory, port=self.port)

    def run(self):
        while True:
            # Receive data from ZeroMQ
            serialized_data = self.pull_socket.recv()
            data = self.deserialize_data(serialized_data)

            print(f"Timestamp delta: {time.time() - data['timestamp']}")

            # Handle data (save to disk, update MPD, etc.)
            self.handle_data(data)

    def handle_data(self, data):
        timestamp = data.pop("timestamp", None)
        segment_duration = data.pop("segment_duration", None)
        frame_rate = data.pop("frame_rate", None)

        for key, item in data.items():
            segment_folder = os.path.join(self.output_directory, "ID{}".format(key))
            segment_path = os.path.join(segment_folder, "segment-{:05d}.bin".format(self.segment_index))
            os.makedirs(segment_folder, exist_ok = True)
            with open(segment_path, "wb") as f:
                pickle.dump(item, f)

            bandwidth = os.path.getsize(segment_path) * 8

            if not self.mpd_manager.initialized:
                self.mpd_manager.add_representation(key, "pointcloud/custom", "unified", bandwidth)

            self.mpd_manager.update_segment(key, "1", segment_path, bandwidth)

        if not self.mpd_manager.initialized:
            self.mpd_manager.initialized = True

        # Update MPD and save it
        self.mpd_manager.update_metadata()
        self.mpd_manager.save_mpd()
        self.segment_index += 1

    def deserialize_data(self, data):
        return pickle.loads(data)


if __name__ == "__main__":
    print("Strarting up")
    # Initialize server
    zmq_address = "tcp://*:5556"
    port = 8080

    server = StreamingServer(zmq_address=zmq_address, port=port)

    threading.Thread(target=server.start_http_server, daemon=True).start()

    # Start streaming server
    server.run()

