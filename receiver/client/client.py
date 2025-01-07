import pickle
import sys
import threading
import time
import requests
import math
import zmq
import yaml
import numpy as np
from queue import Queue
from datetime import datetime
from mpd_parser import MPDParser

LOG = True

shared_epoch = datetime(2024, 1, 1, 0, 0, 0).timestamp()

class StreamingClient:
    def __init__(self, config_file): 
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Configuration
        self.mpd_url = config.get("mpd_url")
        self.max_buffer_size = config.get("client_buffer_size")
        self.fixed_quality = config.get("fixed_quality", None)
        self.bandwidth = config.get("bandwidth", 1000) 
        self.request_offset = config.get("request_offset")
        self.target_fps = config.get("target_fps")
        self.decoder_push_address = config.get("client_push_address")
        self.decoder_pull_address = config.get("client_pull_address")
        self.visualizer_push_address = config.get("visualizer_push_address")

        # MPD Data
        self.segment_duration = None
        self.last_publish_time = None
        self.mpd_data = None
        self.server_active = False

        # Buffers
        self.playout_buffer = Queue(maxsize=self.max_buffer_size)
        self.downloaded_segments = Queue()
        
        # ZMQ setup
        context = zmq.Context()
        self.decoder_push_socket = context.socket(zmq.PUSH)
        self.decoder_push_socket.connect(self.decoder_push_address)

        self.decoder_pull_socket = context.socket(zmq.PULL)
        self.decoder_pull_socket.bind(self.decoder_pull_address)

        self.visualizer_socket = context.socket(zmq.PUSH)
        self.visualizer_socket.connect(self.visualizer_push_address)



    def initilize_stream(self):
        """
        Initialize the stream by trying to get the MPD
        """
        self.parser = MPDParser(self.mpd_url)

        print("Initilaizing Client", flush=True)

        success = False
        while not success: 
            success = self.parser.fetch_mpd()
            print("Waiting for Server to become active", flush=True)
            time.sleep(0.5)

        self.mpd_data = self.parser.parse_mpd()
        self.segment_duration = self.mpd_data['periods'][0]['adaptation_sets'][0]['segment_template']['duration']
        self.last_publish_time = self.mpd_data.get("publishTime")
        self.server_active = True

    def decide_quality(self):
        """Determine quality level based on bandwidth."""
        if self.fixed_quality is not None:
            return self.fixed_quality
        if self.bandwidth > 2000:
            return 0
        elif self.bandwidth > 1000:
            return 1
        return 2

    def download_segment(self, quality, segment_number):
        base_url = self.mpd_url.rsplit('/', 1)[0]
        media_template = self.mpd_data['periods'][0]['adaptation_sets'][0]['segment_template']['media']
        segment_url = base_url + "/" + media_template.replace("$Number$", "{:015d}".format(segment_number))
        segment_url = segment_url.replace("$RepresentationID$", str(quality))

        for attempt in range(3):  # Retry logic
            try:
                response = requests.get(segment_url, stream=True)
                if response.status_code == 200:
                    return response.content
                time.sleep(self.segment_duration / 3)  # Short delay before retry
            except Exception as e:
                print(f"Failed to download segment {segment_number}, attempt {attempt + 1}: {e}")
        return None

    def mpd_updater(self):
        """Periodically checks and updates MPD."""
        while True:
            timestamp = datetime.now().timestamp() 

            success = self.parser.fetch_mpd()
            if not success:
                print("Unsuccesful fetching")
                time.sleep(1.0)
                break

            new_mpd_data = self.parser.parse_mpd()

            if new_mpd_data.get("publishTime") != self.last_publish_time:
                self.mpd_data = new_mpd_data
                self.last_publish_time = new_mpd_data.get("publishTime")

                next_segment_number = math.floor((timestamp - self.request_offset) / self.segment_duration)
                if next_segment_number > self.last_segment_number:
                    self.segment_downloader(next_segment_number)
                    self.last_segment_number = next_segment_number

            sleep_time = max(0, self.segment_duration - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)
            
            if LOG:
                print(f"Sleeping for {sleep_time:.2f} seconds", flush=True)

    def segment_downloader(self, next_segment_number):
        """Downloads and sends segments to the decoder."""
        # This will require the qualities available and the current estimated bandwidth
        quality = self.decide_quality()
        data = self.download_segment(quality, next_segment_number)
        if LOG:
            print("Downloaded segment {}".format(next_segment_number, flush=True))

        if data:
            self.downloaded_segments.put(next_segment_number)
            self.decoder_push_socket.send(data)
        else: 
            print("segment_downloader: Not downloaded...", flush=True)


    def decoder_receiver(self):
        """Receives processed data from the decoder."""
        while True:
            decoded_data = self.decoder_pull_socket.recv()
            self.prepare_for_rendering(decoded_data)


    def visualizer_sender(self):
        """Sends processed data to the visualizer."""
        while True:
            timestamp = datetime.now().timestamp()

            if LOG:
                print("Playoutbufer size: {} Frames, {} sec.".format(
                    self.playout_buffer.qsize(), 
                    self.playout_buffer.qsize() * (1 / self.target_fps)
                    ), flush=True)

            if self.playout_buffer.qsize() > 0:
                data = self.playout_buffer.get()
                self.visualizer_socket.send(data)
            
            # Sleep to maintain target fps
            sleep_time = max(0, (1/self.target_fps) - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)

    def prepare_for_rendering(self, data):
        """
        Unpacks the decoded segment and prepares it for rendering
        """
        data = pickle.loads(data)
        packed_frames = []
        for frame in data:
            points = frame["points"] + 100
            colors = 255 * frame["colors"]
            #timestamp = frame["timestamp"]

            # Pack to bytes
            points_bytes = np.array(points, dtype=np.float32).tobytes()
            colors_bytes = np.array(colors, dtype=np.uint8).tobytes()
            data = points_bytes + colors_bytes

            self.playout_buffer.put(data)

        

    def start(self):
        """Starts all threads."""
        self.initilize_stream()

        self.last_segment_number = 2

        # Start threads
        threading.Thread(target=self.mpd_updater, daemon=True).start()
        threading.Thread(target=self.decoder_receiver, daemon=True).start()
        threading.Thread(target=self.visualizer_sender, daemon=True).start()

        # Keep main thread alive
        while True:
            time.sleep(1)

# Example usage
if __name__ == "__main__":
    client = StreamingClient("./shared/config.yaml")
    client.start()
