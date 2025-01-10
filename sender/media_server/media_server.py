import math
import sys
import os
import yaml
import csv
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

        self.csv_file = None

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
            t_received = time.time()
            data = self.deserialize_data(serialized_data)
            data["sideinfo"]["timestamps"]["media_server_received"] = t_received

            with self.buffer_lock:
                self.segment_buffer.append(data)


    def process_segments(self):
        """
        Process buffered data at fixed time intervals (segment_duration).
        """
        while True:
            #timestamp = datetime.now().timestamp()
            timestamp = time.time()
            current_segment_id= math.floor(timestamp/self.segment_duration)

            # Process and flush the current segment
            if len(self.segment_buffer) > 0:
                with self.buffer_lock:
                    segment = self.segment_buffer.popleft()
            else:
                time.sleep(0.01)
                continue

            self.handle_data(segment, current_segment_id)
            self.cleanup_queue.append(current_segment_id)

            sleep_time = max(0, self.segment_duration - (time.time() - timestamp))
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



    def handle_data(self, segment, timestamp):
        """
        Handle the data
        """
        sideinfo = segment.pop("sideinfo", None)
        data = segment.pop("compressed_data", None)

        segment_number = math.floor((timestamp) / self.segment_duration)
        sideinfo["ID"] = timestamp

        for key in sorted(data):
            item = data[key]
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
                if key == 0:
                    self.mpd_manager.add_representation(key, "pointcloud/custom", "raw", bandwidth)
                else:
                    self.mpd_manager.add_representation(key, "pointcloud/custom", "unified", bandwidth)

            self.mpd_manager.update_segment(key, "1", segment_path, bandwidth)


        if not self.mpd_manager.initialized:
            self.mpd_manager.initialized = True

        # Update MPD and save it
        self.mpd_manager.update_metadata()
        self.mpd_manager.save_mpd()

        sideinfo["timestamps"]["server_published"] = time.time()

        # Log data to file
        self.process_logs_and_save(sideinfo)

    def process_logs_and_save(self, data):
        if self.csv_file is None:
            self.csv_file = "./results/sender/run-{:015d}.csv".format(math.floor(time.time()))

        # Helper function to flatten a nested dictionary
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}_{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)

            # Flatten the input dictionary
        flat_data = flatten_dict(data)

        # If the file doesn't exist, create it with headers
        file_exists = os.path.isfile(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=flat_data.keys())

            # Write headers if the file is new
            if not file_exists:
                writer.writeheader()

            # Append the row to the CSV
            writer.writerow(flat_data)


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
