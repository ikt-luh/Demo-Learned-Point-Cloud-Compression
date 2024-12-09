import time
import os
import zmq
import pickle
from server import HTTPServerHandler
from mpd_manager import MPDManager

class StreamingServer:
    def __init__(self, output_directory="./", zmq_address="tcp://*:5556", port=8080):
        # Setup directories and MPD Manager
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

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
        print("Starting Streaming Server...")
        while True:
            # Receive data from ZeroMQ
            serialized_data = self.pull_socket.recv()
            data = self.deserialize_data(serialized_data)

            print("Received data:")
            print(f"Timestamp delta: {time.time() - data['timestamp']}")
            print(f"Point shape: {data['points'].shape}")

            # Handle data (save to disk, update MPD, etc.)
            self.handle_data(data)

    def handle_data(self, data):
        # Example: Save the data to disk
        segment_path = os.path.join(self.output_directory, f"segment-{self.segment_index}.bin")
        with open(segment_path, "wb") as f:
            pickle.dump(data, f)

        # Update MPD and save it
        self.mpd_manager.update_mpd(self.segment_index, bandwidth=999)
        self.mpd_manager.save_mpd()

        # Increment segment index
        self.segment_index += 1

    @staticmethod
    def deserialize_data(data):
        return pickle.loads(data)


if __name__ == "__main__":
    # Initialize server
    output_directory = "./"
    zmq_address = "tcp://*:5556"
    port = 8080

    server = StreamingServer(output_directory=output_directory, zmq_address=zmq_address, port=port)

    import threading
    threading.Thread(target=server.start_http_server, daemon=True).start()

    # Start streaming server
    server.run()

