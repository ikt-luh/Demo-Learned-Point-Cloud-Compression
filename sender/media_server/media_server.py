import math
import sys
import os
import time
import zmq
import pickle
import threading
from datetime import datetime
from server import HTTPServerHandler
from mpd_manager import MPDManager

shared_epoch = datetime(2024, 1, 1, 0, 0, 0).timestamp()


class StreamingServer:
    def __init__(self, ip_addr, output_directory="./media", zmq_address="tcp://*:5556", port=8080, segment_duration=1.0):
        # Setup directories and MPD Manager
        self.ip_addr = ip_addr
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.mpd_manager = MPDManager(self.output_directory)
        self.mpd_manager.setup_adaptation_set()

        # Setup ZeroMQ
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(zmq_address)

        self.port = port
        self.segment_duration = segment_duration

        # Buffer for segment data
        self.segment_buffer = []  # {segment_number: [data1, data2, ...]}
        self.lock = threading.Lock()

    def start_http_server(self):
        """
        Start the HTTP Server at the output directory defined in the constructor.
        """
        HTTPServerHandler.start(directory=self.output_directory, ip_addr=self.ip_addr, port=self.port)

    def run(self):
        """
        Run the media server until termination. 
        Receive serialized data, deserialize and buffer it for processing in intervals.
        """
        # Start a thread to process data at fixed intervals
        threading.Thread(target=self.process_segments, daemon=True).start()
        threading.Thread(target=self.cleanup_segments, daemon=True).start()

        while True:
            serialized_data = self.pull_socket.recv()
            print("Received {}".format(time.time()))
            sys.stdout.flush()
            data = self.deserialize_data(serialized_data)
            with self.lock:
                self.segment_buffer.append(data)


    def process_segments(self):
        """
        Process buffered data at fixed time intervals (segment_duration).
        """
        # Startup buffer filling
        while len(self.segment_buffer) < 1:
            time.sleep(0.1)

        while True:
            timestamp = datetime.now().timestamp()
            current_segment = math.floor(timestamp)
            next_segment = current_segment + 1
            next_segment_time = shared_epoch + next_segment * self.segment_duration

            # Process and flush the current segment
            with self.lock:
                if len(self.segment_buffer) > 0:
                    data = self.segment_buffer.pop(0)
                    threading.Thread(target=self.handle_data, args=(data, current_segment)).start()
                else:
                    print("empty buffer")

            current_segment = next_segment
        
            sleep_time = max(0, self.segment_duration - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)
            print("Running")

    def cleanup_segments(self):
        num_reps = 3 # TODO
        while True:
            timestamp = datetime.now().timestamp()
            number = math.floor((timestamp) / self.segment_duration) - 10
            for key in range(num_reps):
                segment_folder = os.path.join(self.output_directory, f"ID{key}")
                segment_path = os.path.join(segment_folder, f"segment-{number:015d}.bin")
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                sys.stdout.flush()

            sleep_time = max(0, self.segment_duration - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)


    def handle_data(self, data, timestamp):
        """
        Handle the data
        """
        _ = data.pop("timestamp", None)
        segment_duration = data.pop("segment_duration", None)
        frame_rate = data.pop("frame_rate", None)

        number = math.floor((timestamp) / self.segment_duration)

        for key, item in data.items():
            segment_folder = os.path.join(self.output_directory, f"ID{key}")
            segment_path = os.path.join(segment_folder, f"segment-{number:015d}.bin")
            tmp_segment_path = os.path.join(segment_folder, f"segment-{number:015d}_tmp.bin")
            os.makedirs(segment_folder, exist_ok=True)

            # Write the segment to disk
            with open(tmp_segment_path, "wb") as f:
                pickle.dump(item, f)
            os.rename(tmp_segment_path, segment_path)

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

        Parameters:
            data (string): Serialized data from pickle

        Returns:
            data (dict): Deserialized data
        """
        return pickle.loads(data)


if __name__ == "__main__":
    # Initialize server
    ip_addr = "0.0.0.0"
    zmq_address = "tcp://*:5556"
    port = 8080
    segment_duration = 1.0  # Set segment duration in seconds

    server = StreamingServer(ip_addr, zmq_address=zmq_address, port=port, segment_duration=segment_duration)

    threading.Thread(target=server.start_http_server, daemon=True).start()

    # Start streaming server
    server.run()
