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
from downloader import SegmentDownloader
from gui import create_flask_app

LOG = True

shared_epoch = datetime(2024, 1, 1, 0, 0, 0).timestamp()

class StreamingClient:
    def __init__(self, config_file): 
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Configuration
        self.mpd_url = config.get("mpd_url")
        self.max_buffer_size = config.get("client_buffer_size")
        self.request_offset = config.get("request_offset")
        self.target_fps = config.get("target_fps")
        self.decoder_push_address = config.get("client_push_address")
        self.decoder_pull_address = config.get("client_pull_address")
        self.visualizer_push_address = config.get("visualizer_push_address")

        fixed_quality_mode = config.get("fixed_quality_mode", True)
        init_quality = config.get("init_quality", 0)

        # MPD Data
        self.segment_duration = None
        self.last_publish_time = None
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

        # Logic
        self.segment_downloader = SegmentDownloader(fixed_quality_mode, init_quality)
        self.mpd_parser = MPDParser(self.mpd_url)
 

    def download_loop(self):
        """Periodically checks and updates MPD."""
        while True:
            timestamp = datetime.now().timestamp() 

            while not self.mpd_parser.update_mpd():
                print("Waiting for MPD to become available", flush=True)

            segment_duration = self.mpd_parser.get_segment_duration()
            publish_time = self.mpd_parser.get_publish_time()

            if publish_time != self.last_publish_time:
                self.last_publish_time = publish_time
                next_segment_number = math.floor((timestamp - self.request_offset) / segment_duration)

                if next_segment_number > self.last_segment_number:
                    self.download_segment(next_segment_number)
                    self.last_segment_number = next_segment_number

            sleep_time = max(0, segment_duration - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)
            

    def download_segment(self, next_segment_number):
        """Downloads and sends segments to the decoder."""
        base_url = self.mpd_url.rsplit('/', 1)[0]
        media_template = self.mpd_parser.get_media_template()

        data = self.segment_downloader.download_segment(base_url, media_template, next_segment_number)
        codec_info = self.mpd_parser.get_codec_info(self.segment_downloader.current_quality)

        if data:
            print("Downloaded segment {}".format(next_segment_number), flush=True)
            self.downloaded_segments.put(next_segment_number)
            self.decoder_push_socket.send(pickle.dumps([codec_info, data]))
        else: 
            print("segment_downloader: Not downloaded...", flush=True)


    def decoder_receiver(self):
        """Receives processed data from the decoder."""
        while True:
            decoded_data = self.decoder_pull_socket.recv()
            data = pickle.loads(decoded_data)

            for frame in data:
                points = frame["points"] + 100
                colors = 255 * frame["colors"]
                #timestamp = frame["timestamp"]

                # Pack to bytes
                points_bytes = np.array(points, dtype=np.float32).tobytes()
                colors_bytes = np.array(colors, dtype=np.uint8).tobytes()
                data = points_bytes + colors_bytes

                self.playout_buffer.put(data)


    def visualizer_sender(self):
        """Sends processed data to the visualizer."""
        while True:
            timestamp = datetime.now().timestamp()
            """
            if LOG:
                print("Playoutbufer size: {} Frames, {} sec.".format(self.playout_buffer.qsize(), self.playout_buffer.qsize() * (1 / self.target_fps)), flush=True)
            """
            while self.playout_buffer.empty():
                print("Stalling", flush=True)
                time.sleep(0.1)
                timestamp = datetime.now().timestamp()

            data = self.playout_buffer.get()
            self.visualizer_socket.send(data)
            
            sleep_time = max(0, (1/self.target_fps) - (datetime.now().timestamp() - timestamp))
            time.sleep(sleep_time)

        
    def start(self):
        """Starts all threads."""
        #self.initilize_stream()

        self.last_segment_number = 0

        # Start threads
        threading.Thread(target=self.download_loop, daemon=True).start()
        threading.Thread(target=self.decoder_receiver, daemon=True).start()
        threading.Thread(target=self.visualizer_sender, daemon=True).start()
        
        # GUI
        gui = create_flask_app(self)
        threading.Thread(target=lambda: gui.run(host="0.0.0.0", port=5000), daemon=True).start()

        while True:
            time.sleep(1)

if __name__ == "__main__":
    client = StreamingClient("./shared/config.yaml")
    client.start()