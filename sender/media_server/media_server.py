import math
import sys
import os
import yaml
import time
import zmq
import pickle
import threading
from collections import deque
from datetime import datetime
from server import HTTPServerHandler
from mpd_manager import MPDManager

shared_epoch = datetime(2024, 1, 1, 0, 0, 0).timestamp()


class StreamingServer:
    def __init__(self, config_file=None):
        # Load settings from YAML if a file is provided
        if config_file:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
        else:
            config = {}
        # Setup directories and MPD Manager
        self.ip_addr = config.get("ip_addr")
        self.port = config.get("port")
        self.output_directory = config.get("output_directory")
        self.segment_duration = config.get("segment_duration")
        self.pull_address = config.get("media_server_pull_address")

        os.makedirs(self.output_directory, exist_ok=True)

        self.mpd_manager = MPDManager(self.output_directory)
        self.mpd_manager.setup_adaptation_set()

        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind(self.pull_address)

        self.segment_buffer = deque()
        self.buffer_lock = threading.Lock()
        self.io_lock = threading.Lock()
        
        self.cleanup_queue = deque()

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
            print("Received {}".format(time.time()), flush=True)
            data = self.deserialize_data(serialized_data)

            with self.buffer_lock:
                self.segment_buffer.append(data)
            time.sleep(self.segment_duration/2)


    def process_segments(self):
        """
        Process buffered data at fixed time intervals (segment_duration).
        """
        while True:
            timestamp = datetime.now().timestamp()
            current_segment = math.floor(timestamp/self.segment_duration)

            # Process and flush the current segment
            if len(self.segment_buffer) > 0:
                with self.buffer_lock:
                    data = self.segment_buffer.popleft()
            else:
                time.sleep(0.2)
                continue

            #threading.Thread(target=self.handle_data, args=(data, current_segment)).start()
            self.handle_data(data, current_segment)
            self.cleanup_queue.append(current_segment)

            sleep_time = max(0, self.segment_duration - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)


    def cleanup_segments(self):
        num_reps = 3 # TODO
        while True:
            if len(self.cleanup_queue) > 10:
                old_segment_number = self.cleanup_queue.popleft()
                for key in range(num_reps):
                    segment_folder = os.path.join(self.output_directory, f"ID{key}")
                    segment_path = os.path.join(segment_folder, f"segment-{old_segment_number:015d}.bin")
                    if os.path.exists(segment_path):
                        os.remove(segment_path)

            else:
                time.sleep(1)



    def handle_data(self, data, timestamp):
        """
        Handle the data
        """
        _ = data.pop("timestamp", None)
        _ = data.pop("segment_duration", None)
        _ = data.pop("frame_rate", None)

        segment_number = math.floor((timestamp) / self.segment_duration)

        for key, item in data.items():
            segment_folder = os.path.join(self.output_directory, f"ID{key}")
            segment_path = os.path.join(segment_folder, f"segment-{segment_number:015d}.bin")
            tmp_segment_path = os.path.join(segment_folder, f"segment-{segment_number:015d}_tmp.bin")
            os.makedirs(segment_folder, exist_ok=True)

            # Write the segment to disk
            with open(tmp_segment_path, "wb") as f:
                pickle.dump(item, f)
            os.rename(tmp_segment_path, segment_path)
            print("Wrote to {}".format(segment_path), flush=True)

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
    server = StreamingServer(config_file="./shared/config.yaml")

    threading.Thread(target=server.start_http_server, daemon=True).start()

    # Start streaming server
    server.run()
