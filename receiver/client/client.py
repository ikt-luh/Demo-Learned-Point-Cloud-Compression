import requests
import xml.etree.ElementTree as ET
import queue
import threading
import time
import zmq


class StreamingClient:
    def __init__(self, mpd_url, target_buffer_duration=2, max_buffer_duration=10, bandwidth=None):
        self.mpd_parser = MPDParser(mpd_url)
        self.target_buffer_duration = target_buffer_duration
        self.max_buffer_duration = max_buffer_duration
        self.bandwidth = bandwidth

        self.segment_buffer = queue.Queue()
        self.buffer_lock = threading.Lock()
        self.segment_duration = None

        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.connect("tcp://decoder:5555")

    def calculate_buffer_size(self, duration):
        return max(1, int(duration / self.segment_duration))

    def fetch_segment_duration(self):
        first_rep = next(iter(self.mpd_parser.representations.values()))
        self.segment_duration = 1  # Assuming 1 second segments if not specified

    def buffer_segments(self, rep_id):
        target_size = self.calculate_buffer_size(self.target_buffer_duration)
        max_size = self.calculate_buffer_size(self.max_buffer_duration)

        while self.segment_buffer.qsize() < max_size:
            try:
                self.buffer_lock.acquire()
                segment_data, metadata = self.download_segment(rep_id)
                self.segment_buffer.put((segment_data, metadata))
                print(f"Buffered segment {metadata['segment_number']} for {rep_id}. Buffer size: {self.segment_buffer.qsize()}")

                if self.segment_buffer.qsize() >= target_size:
                    break
            finally:
                self.buffer_lock.release()

    def download_segment(self, rep_id):
        rep_data = self.mpd_parser.representations[rep_id]
        segment_number = rep_data['next_segment']
        segment_url = self.mpd_parser.mpd_url.rsplit('/', 1)[0] + '/' + rep_data['media_template'].replace('$RepresentationID$', rep_id).replace('$Number$', str(segment_number))
        
        print(f"Downloading segment {segment_number} for {rep_id}...")
        response = requests.get(segment_url)
        response.raise_for_status()

        metadata = {
            'type': 'segment',
            'rep_id': rep_id,
            'segment_number': segment_number
        }
        
        self.mpd_parser.representations[rep_id]['next_segment'] += 1
        return response.content, metadata

    def consume_buffer(self):
        while True:
            if not self.segment_buffer.empty():
                self.buffer_lock.acquire()
                try:
                    segment_data, metadata = self.segment_buffer.get()
                    self.send_data(segment_data, metadata)
                    print(f"Consumed segment {metadata['segment_number']} for {metadata['rep_id']}. Buffer size: {self.segment_buffer.qsize()}")
                finally:
                    self.buffer_lock.release()
            time.sleep(0.1)

    def send_data(self, segment_data, metadata):
        self.zmq_socket.send_json(metadata)
        self.zmq_socket.send(segment_data)

    def select_representation(self):
        if self.bandwidth:
            suitable_reps = sorted(
                self.mpd_parser.representations.items(),
                key=lambda x: abs(x[1]['bandwidth'] - self.bandwidth)
            )
            return suitable_reps[0][0]
        return next(iter(self.mpd_parser.representations))

    def start_streaming(self):
        threading.Thread(target=self.consume_buffer, daemon=True).start()

        while True:
            try:
                print("Fetching and parsing MPD...")
                mpd_root = self.mpd_parser.fetch_mpd()
                self.mpd_parser.parse_mpd(mpd_root)

                if not self.segment_duration:
                    self.fetch_segment_duration()

                rep_id = self.select_representation()
                print(f"Selected representation: {rep_id}")

                self.buffer_segments(rep_id)
                time.sleep(1)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(2)


if __name__ == "__main__":
    MPD_URL = "http:/***/media/manifest.mpd"
    target_buffer = 2
    max_buffer = 10
    bandwidth = 10000

    client = StreamingClient(MPD_URL, target_buffer, max_buffer, bandwidth=bandwidth)
    client.start_streaming()