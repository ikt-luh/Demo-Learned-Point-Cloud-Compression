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
from shared.file_utils import process_logs_and_save

LOG = True

shared_epoch = datetime(2024, 1, 1, 0, 0, 0).timestamp()

class StreamingClient:
    def __init__(self, config_file): 
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Configuration
        self.mpd_url = config.get("mpd_url")
        self.request_offset = config.get("request_offset")
        self.playout_offset = config.get("playout_offset", 8)
        self.decoder_push_address = config.get("client_push_address")
        self.decoder_pull_address = config.get("client_pull_address")
        self.visualizer_push_address = config.get("visualizer_push_address")

        fixed_quality_mode = config.get("fixed_quality_mode", True)
        init_quality = config.get("init_quality", 2)

        # MPD Data
        self.segment_duration = None
        self.last_publish_time = None
        self.server_active = False

        # GUI Info
        self.current_bandwidth = 0
        self.current_num_points = 0
        self.current_latencies = {
            "e1": 0, "e2": 0, "e3": 0, "e4": 0, "e5": 0, "e6": 0, "e7": 0,
        }

        # Buffers
        self.playout_buffer = Queue()
        self.playout_time_buffer = Queue()
        self.sideinfo_queue = Queue()
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

        self.csv_file = None
 

    def download_loop(self):
        """Periodically checks and updates MPD."""
        while True:
            while not self.mpd_parser.update_mpd():
                print("Waiting for MPD to become available", flush=True)

            segment_duration = self.mpd_parser.get_segment_duration()
            publish_time = self.mpd_parser.get_publish_time()

            timestamp = time.time()
            if publish_time != self.last_publish_time:
                self.last_publish_time = publish_time
                next_segment_number = math.floor(timestamp  / segment_duration)

                if next_segment_number > self.last_segment_number:
                    self.download_segment(next_segment_number)
                    self.last_segment_number = next_segment_number

                wake_up_time = (next_segment_number + 1) * segment_duration - self.request_offset
                sleep_time = max(0, wake_up_time - time.time())
                time.sleep(sleep_time)
            else:
                time.sleep(0.3)
            

    def download_segment(self, next_segment_number):
        """Downloads and sends segments to the decoder."""
        base_url = self.mpd_url.rsplit('/', 1)[0]
        media_template = self.mpd_parser.get_media_template()

        data = self.segment_downloader.download_segment(base_url, media_template, next_segment_number)
        codec_info = self.mpd_parser.get_codec_info(self.segment_downloader.current_quality)
        bandwidth = self.mpd_parser.get_bandwidth(self.segment_downloader.current_quality)

        if data:
            (data, sideinfo) = pickle.loads(data)
            sideinfo["ID"] = next_segment_number
            sideinfo["quality"] = self.segment_downloader.current_quality
            sideinfo["codec_info"] = codec_info
            sideinfo["timestamps"]["client_received"] = time.time()

            segment = {"data": pickle.dumps(data), "sideinfo": sideinfo}

            self.downloaded_segments.put(next_segment_number)
            self.decoder_push_socket.send(pickle.dumps(segment))

            print("Downloaded segment {}".format(next_segment_number), flush=True)
        else: 
            print("segment_downloader: Not downloaded...", flush=True)




    def decoder_receiver(self):
        """Receives processed data from the decoder."""
        while True:
            decoded_data = self.decoder_pull_socket.recv()
            segment = pickle.loads(decoded_data)
            data = segment["data"]
            sideinfo = segment["sideinfo"]
            segment_start_time = max(sideinfo["ID"] + self.playout_offset, time.time())
            sideinfo["timestamps"]["playout"] = []

            num_frames = len(data)
            points_per_segment = 0
            for i, frame in enumerate(data):
                points = frame["points"] + 100
                colors = 255 * frame["colors"]

                # Pack to bytes
                points_bytes = np.array(points, dtype=np.float32).tobytes()
                colors_bytes = np.array(colors, dtype=np.uint8).tobytes()
                data = points_bytes + colors_bytes

                # Compute playout time
                next_playout_time = segment_start_time + ((i + 1) / num_frames)
                self.playout_buffer.put(data)
                self.playout_time_buffer.put(next_playout_time)

                sideinfo["timestamps"]["playout"].append(next_playout_time)
                points_per_segment += points.shape[0]

            if self.csv_file is None:
                self.csv_file = "./evaluation/logs/receiver/{:015d}.csv".format(math.floor(time.time()))

            # GUI updates
            quality = sideinfo["quality"]
            self.current_bandwidth = sideinfo["gop_info"]["bandwidth"][quality] / 1000
            self.current_num_points = points_per_segment
            self.current_latencies = {
                "e1": sideinfo["enc_time_measurements"]["analysis"],
                "e2": sideinfo["enc_time_measurements"]["hyper_analysis"],
                "e3": sideinfo["enc_time_measurements"]["factorized_model"],
                "e4": sideinfo["enc_time_measurements"]["hyper_synthesis"],
                "e5": sideinfo["enc_time_measurements"]["gaussian_model"],
                "e6": sideinfo["enc_time_measurements"]["geometry_compression"],
                "e7": sum(sideinfo["enc_time_measurements"]["bitstream_writing"]),
                "d1": sideinfo["time_measurements"]["bitstream_reading"],
                "d2": sideinfo["time_measurements"]["geometry_decompression"],
                "d3": sideinfo["time_measurements"]["factorized_model"],
                "d4": sideinfo["time_measurements"]["hyper_synthesis"],
                "d5": sideinfo["time_measurements"]["guassian_model"],
                "d6": sideinfo["time_measurements"]["synthesis_transform"],
            }

            # Logs
            process_logs_and_save(sideinfo, self.csv_file)


    def visualizer_sender(self):
        """Sends processed data to the visualizer."""
        counter = 0 
        while True:
            while self.playout_buffer.empty():
                print("Stalling", flush=True)
                time.sleep(0.05)

            # Send Frame
            frame = self.playout_buffer.get()
            self.visualizer_socket.send(frame)

            # Sleep until next playout
            playout_time = self.playout_time_buffer.get()


            sleep_time = max(0, playout_time - time.time())

            if sleep_time <= 0:
                print("Catching up", flush=True)

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
        gui, socket = create_flask_app(self)
        threading.Thread(target=lambda: gui.run(host="0.0.0.0", port=5000), daemon=True).start()

        while True:
            time.sleep(1)
    
    def get_bandwidth(self):
        return self.current_bandwidth

    def get_latencies(self):
        return self.current_latencies

    def get_num_points(self):
        return self.current_num_points

if __name__ == "__main__":
    client = StreamingClient("./shared/config.yaml")
    client.start()