import time
import yaml
import threading
import queue
import zmq
import pickle
import numpy as np
import torch

class Encoder:
    def __init__(self, max_queue_size=60, target_fps=3, gop_size=3, segment_length=2.0):
        context = zmq.Context()

        # ZeroMQ
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect("tcp://mediaserver:5556")

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5555")

        # Bounded queue for frame buffering
        self.queue = queue.Queue(maxsize=max_queue_size)

        # GoP and Framerate Handling
        self.target_fps = target_fps
        self.gop_size = gop_size
        self.segment_length = segment_length

        # Thread-safe queue for batches
        self.batch_queue = queue.Queue()

        # Start the processing worker thread
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def run(self):
        batch = []
        start_time_stamp = None

        while True:
            # Receive and deserialize incoming data
            data = pickle.loads(self.pull_socket.recv())
            time_stamp = data["timestamp"]

            # Initialize the start time stamp for the first batch
            if start_time_stamp is None:
                start_time_stamp = time_stamp

            # Calculate current segment length
            curr_segment_length = time_stamp - start_time_stamp

            if curr_segment_length <= self.segment_length:
                # Add to batch if within the segment
                batch.append(data)
            else:
                # Push the batch to the processing queue
                self.batch_queue.put(batch)

                # Reset batch and adjust start time
                start_time_stamp += self.segment_length  # Ensure fixed intervals
                batch = [data]

    def worker(self):
        while True:
            # Wait for a batch to process
            batch = self.batch_queue.get()
            if batch is None:
                break  # Exit worker thread if None is received
            self.process(batch)

    def process(self, batch):
        # Sampling
        sampled_batch = self.sample(batch)

        # Compression
        compressed_data = self.compress(sampled_batch)

        # Serialization
        serialized_data = self.serialize_data(compressed_data)

        # Sending
        self.push_socket.send(serialized_data)

    def sample(self, batch):
        """
        Uniformly samples n frames from a list of data with arbitrary number of frames based on timestamps.

        Parameters:
            batch (list of dict): List of dictionaries, each containing a "timestamp" key.
        
        Returns:
            sampled_batch (list of dict): Sampled frames.
        """
        timestamps = [item["timestamp"] for item in batch]
        start_time = timestamps[0]
        n = int(self.segment_length * self.target_fps)  # Total number of frames to sample
        step = self.segment_length / n

        target_timestamps = [start_time + i * step for i in range(n)]

        sampled_batch = []
        for target in target_timestamps:
            closest_frame = min(batch, key=lambda item: abs(item["timestamp"] - target))
            sampled_batch.append(closest_frame)

        print(sampled_batch[0]["timestamp"])
        print(sampled_batch[-1]["timestamp"])
        print("")
        return sampled_batch

    def compress(self, sampled_batch):
        """
        Mock compression function. Currently, just returns the input.
        """
        return sampled_batch  # Replace with actual compression logic

    def serialize_data(self, data):
        """
        Serializes data using pickle.
        """
        return pickle.dumps(data)

            
if __name__ == "__main__":
    enc = Encoder()
    enc.run()