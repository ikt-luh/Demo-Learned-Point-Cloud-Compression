import time
import gzip
import json
import os
import requests
from lxml import etree
from queue import Queue
import threading
import random


class MPDParser:
    def __init__(self, mpd_url):
        self.mpd_url = mpd_url
        self.mpd_data = None
        self.qualities = []

    def fetch_mpd(self):
        """Fetch the MPD file from the server."""
        print(f"Fetching MPD from {self.mpd_url}")
        response = requests.get(self.mpd_url)
        response.raise_for_status()
        return response.content

    def parse_mpd(self):
        """Parse the MPD and extract qualities and segment details."""
        self.mpd_data = self.fetch_mpd()
        root = etree.fromstring(self.mpd_data)
        namespaces = {"": "urn:mpeg:dash:schema:mpd:2011"}

        adaptation_set = root.find(".//AdaptationSet", namespaces)
        segment_template = adaptation_set.find(".//SegmentTemplate", namespaces)

        media_template = segment_template.attrib["media"]
        start_number = int(segment_template.attrib["startNumber"])
        timescale = int(segment_template.attrib["timescale"])
        duration = int(segment_template.attrib["duration"])

        # Collect all qualities
        self.qualities = [
            rep.attrib["id"] for rep in adaptation_set.findall(".//Representation", namespaces)
        ]

        # Return segment parameters
        return {
            "media_template": media_template,
            "start_number": start_number,
            "timescale": timescale,
            "duration": duration,
        }


class BandwidthEstimator:
    def estimate_bandwidth(self):
        """Simulate network bandwidth estimation."""
        return random.uniform(0.5, 3.0)  # Simulate bandwidth between 0.5 Mbps and 3.0 Mbps


class BitrateAdaptation:
    def __init__(self):
        self.bandwidth_map = {
            "V100": 0.5,  # Requires 0.5 Mbps
            "V200": 1.0,  # Requires 1 Mbps
            "V300": 2.0,  # Requires 2 Mbps
        }

    def select_quality(self, bandwidth):
        """Select the best quality based on available bandwidth."""
        suitable_qualities = [q for q, b in self.bandwidth_map.items() if b <= bandwidth]
        return suitable_qualities[-1] if suitable_qualities else "V100"


class SegmentDownloader:
    def __init__(self, base_url):
        self.base_url = base_url

    def download_segment(self, quality, segment_number, media_template):
        """Download a segment."""
        segment_file = media_template.replace("$RepresentationID$", quality).replace(
            "$Number%05d$", f"{segment_number:05d}"
        )
        segment_url = f"{self.base_url}/{segment_file}"

        try:
            response = requests.get(segment_url, stream=True)
            response.raise_for_status()
            segment_data = gzip.decompress(response.content).decode('utf-8')
            return json.loads(segment_data)
        except Exception as e:
            print(f"Error downloading segment {segment_file}: {e}")
            return None


class SegmentBuffer:
    def __init__(self, buffer_size):
        self.segment_buffer = Queue(maxsize=buffer_size)

    def add_to_buffer(self, segment):
        """Add a segment to the buffer."""
        if not self.segment_buffer.full():
            self.segment_buffer.put(segment)
            return True
        return False

    def get_from_buffer(self):
        """Get a segment from the buffer."""
        if not self.segment_buffer.empty():
            return self.segment_buffer.get()
        return None


class StreamingClient:
    def __init__(self, base_url, mpd_url, output_dir, segment_duration=0.5, buffer_size=3):
        self.base_url = base_url
        self.output_dir = output_dir
        self.segment_duration = segment_duration

        # Initialize modular components
        self.mpd_parser = MPDParser(mpd_url)
        self.bandwidth_estimator = BandwidthEstimator()
        self.bitrate_adaptation = BitrateAdaptation()
        self.segment_downloader = SegmentDownloader(base_url)
        self.segment_buffer = SegmentBuffer(buffer_size)

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def stream(self):
        """Start the streaming process with producer and consumer threads."""
        mpd_info = self.mpd_parser.parse_mpd()
        media_template = mpd_info["media_template"]
        start_number = mpd_info["start_number"]
        duration = mpd_info["duration"] / mpd_info["timescale"]

        # Start producer and consumer threads
        producer_thread = threading.Thread(target=self.producer, args=(media_template, start_number, duration))
        consumer_thread = threading.Thread(target=self.consumer)
        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

    def producer(self, media_template, start_number, duration):
        """Download segments and add them to the buffer."""
        segment_index = start_number
        while True:
            bandwidth = self.bandwidth_estimator.estimate_bandwidth()  # Estimate bandwidth
            current_quality = self.bitrate_adaptation.select_quality(bandwidth)  # Select quality based on bandwidth
            print(current_quality)
            print(f"Estimated bandwidth: {bandwidth:.2f} Mbps, selected quality: {current_quality}")

            segment = self.segment_downloader.download_segment(current_quality, segment_index, media_template)
            if segment:
                if self.segment_buffer.add_to_buffer(segment):
                    print(f"Buffered segment {segment_index} at quality {current_quality}")
                segment_index += 1
            time.sleep(duration)  # Sleep for the segment duration to simulate downloading interval

    def consumer(self):
        """Consume segments from the buffer for playback."""
        while True:
            segment = self.segment_buffer.get_from_buffer()  # Get segment from buffer
            if segment:
                print(f"Playing segment: {segment}")
                time.sleep(self.segment_duration)  # Simulate playback time (segment duration)
