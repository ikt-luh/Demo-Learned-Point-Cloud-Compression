import time
import requests
import zmq
from threading import Thread

class StreamingClient:
    def __init__(self, mpd_url, zmq_address):
        self.mpd_url = mpd_url
        self.zmq_address = zmq_address
        self.buffer = []
        self.buffer_lock = Thread()
        self.max_buffer_size = 10  # Can be adjusted based on minBufferTime and segment duration
        self.buffer_target = 2
        self.mpd_data = None
        self.segment_duration = None
        self.last_publish_time = None

        self.next_segment = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(self.zmq_address)

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
        segment_url = base_url + "/" + media_template.replace("$Number$", str(next_segment_number))
        segment_url = base_url + "/" + media_template.replace("$RepresentationID$", str(quality))
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to fetch segment: HTTP {response.status_code}")

    def fill_buffer(self):
        while True:
            t_0 = time.time()
            if len(self.buffer) < self.max_buffer_size:
                try:
                    next_segment_number = math.floor((time.time() - self.buffer_target) / self.segment_duration)
                    segment_data = self.download_segment(segment_index)
                    self.buffer.append(segment_data)
                except Exception as e:
                    print("Not available")
                    time.sleep(0.1)

            if len(self.buffer) * self.segment_duration > self.buffer_target:
                t_1 = time.time()
                passed_time = t_1 - t_0
                time.sleep(self.segment_duration - passed_time)

    def consume_buffer(self):
        while True:
            t_0 = time.time()
            if self.buffer:
                segment_data = self.buffer.pop(0)
                self.socket.send(segment_data)
            t_1 = time.time()
            passed_time = t_1 - t_0
            time.sleep(self.segment_duration - passed_time)

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
    mpd_url = "<URL_TO_MPD_FILE>"  # Replace with actual MPD URL
    zmq_address = "tcp://127.0.0.1:5555"  # Example ZMQ address

    client = StreamingClient(mpd_url, zmq_address)
    client.start()
