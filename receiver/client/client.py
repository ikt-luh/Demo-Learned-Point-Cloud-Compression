from datetime import datetime
import math
import time
import requests
import zmq
from threading import Thread

from mpd_parser import MPDParser

shared_epoch = datetime(2024, 1, 1, 0, 0, 0).timestamp()

class StreamingClient:
    def __init__(self, mpd_url, zmq_address):
        self.mpd_url = mpd_url
        self.zmq_address = zmq_address
        self.buffer = []
        self.buffer_lock = Thread()
        self.max_buffer_size = 10  # Can be adjusted based on minBufferTime and segment duration
        self.buffer_target = 3
        self.mpd_data = None
        self.segment_duration = None
        self.last_publish_time = None

        self.next_segment = None

        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect("tcp://decoder:5555")

    def fetch_and_parse_mpd(self):
        parser = MPDParser(self.mpd_url)
        parser.fetch_mpd()
        self.mpd_data = parser.parse_mpd()
        self.segment_duration = self.mpd_data['periods'][0]['adaptation_sets'][0]['segment_template']['duration'] 
        self.last_publish_time = self.mpd_data.get("publishTime")

        self.next_segment = 0

    def download_segment(self, quality, segment_number):
        base_url = self.mpd_url.rsplit('/', 1)[0]
        media_template = self.mpd_data['periods'][0]['adaptation_sets'][0]['segment_template']['media']
        segment_url = base_url + "/" + media_template.replace("$Number$", "{:015d}".format(segment_number))
        segment_url = segment_url.replace("$RepresentationID$", str(quality))
        response = requests.get(segment_url, stream=True)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to fetch segment: HTTP {response.status_code}")

    def fill_buffer(self):
        last_segment = 0
        while True:
            t_0 = time.time()
            if len(self.buffer) < self.max_buffer_size:
                try:
                    timestamp = datetime.now().timestamp() - shared_epoch
                    next_segment_number = math.floor((timestamp - self.buffer_target) / self.segment_duration)
                    if next_segment_number > last_segment:
                        segment_data = self.download_segment(0, next_segment_number)
                        last_segment = next_segment_number
                except Exception as e:
                    time.sleep(0.5)

            if len(self.buffer) * self.segment_duration > self.buffer_target:
                t_1 = time.time()
                passed_time = t_1 - t_0
                time.sleep(self.segment_duration - passed_time)

    def consume_buffer(self):
        next_target_time = time.time()
        while True:
            current_time = time.time()
            if self.buffer:
                print("{}: Consuming buffer ....".format(current_time))
                segment_data = self.buffer.pop(0)
                self.socket.send(segment_data)
                
                next_target_time += self.segment_duration

                sleep_time = max(0, next_target_time - time.time())
                time.sleep(sleep_time)
            else:
                time.sleep(0.05) #Wait for buffer filling

    def check_for_updates(self):
        while True:
            time.sleep(float(self.mpd_data.get("minimumUpdatePeriod", 2)))
            parser = MPDParser(self.mpd_url)
            parser.fetch_mpd()
            new_mpd_data = parser.parse_mpd()

            if new_mpd_data.get("publishTime") != self.last_publish_time:
                self.mpd_data = new_mpd_data
                self.last_publish_time = new_mpd_data.get("publishTime")

    def start(self):
        self.fetch_and_parse_mpd()

        Thread(target=self.fill_buffer).start()
        Thread(target=self.consume_buffer).start()
        if self.mpd_data["type"] == "dynamic":
            Thread(target=self.check_for_updates).start()

# Example usage
if __name__ == "__main__":
    mpd_url = "http://172.23.181.103:8080/media/manifest.mpd"  # Replace with actual MPD URL
    zmq_address = "tcp://decoder:5555"  # Example ZMQ address

    client = StreamingClient(mpd_url, zmq_address)
    client.start()
