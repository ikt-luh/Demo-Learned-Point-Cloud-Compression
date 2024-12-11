import math
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

        self.mpd_manager = MPDManager(self.output_directory)
        self.mpd_manager.setup_adaptation_set()

        # Setup ZeroMQ
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(zmq_address)

        self.port = port

    def start_http_server(self):
        """
        Start the HTTP Server at the output directory defined in the constructor.
        """
        HTTPServerHandler.start(directory=self.output_directory, port=self.port)

    def run(self):
        """
        Run the media server until termination. 
        Receive serialized data, deserialize and handle data (i.e. updating MPD, saving to disk)
        """
        while True:
            serialized_data = self.pull_socket.recv()
            data = self.deserialize_data(serialized_data)

            self.handle_data(data)
            #print(f"Timestamp delta: {time.time() - data['timestamp']}")

    def handle_data(self, data):
        """
        Handle the data
        """
        timestamp = data.pop("timestamp", None)
        segment_duration = data.pop("segment_duration", None)
        frame_rate = data.pop("frame_rate", None)

        number = math.floor(time.time() / segment_duration)
        for key, item in data.items():
            segment_folder = os.path.join(self.output_directory, "ID{}".format(key))
            segment_path = os.path.join(segment_folder, "segment-{:015d}.bin".format(number))
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

    def deserialize_data(self, data):
        """
        Deserialize the data.

        Paramters:
            data (string) : Serialized data from pickle
        Returns:
            data (dict) : Deserialized data
        """

        return pickle.loads(data)


if __name__ == "__main__":
    # Initialize server
    zmq_address = "tcp://*:5556"
    port = 8080

    server = StreamingServer(zmq_address=zmq_address, port=port)

    threading.Thread(target=server.start_http_server, daemon=True).start()

    # Start streaming server
    server.run()

