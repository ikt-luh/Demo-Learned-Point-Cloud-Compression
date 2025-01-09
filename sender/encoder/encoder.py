import sys
import time
import yaml
import threading
import queue
import zmq
import pickle
import numpy as np
import torch
import concurrent.futures

#from codec_single import CompressionPipeline
from codec_pipeline import CompressionPipeline

class Encoder:
    def __init__(self, config_file=None): 
        # Load settings from YAML if a file is provided
        if config_file:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
        else:
            config = {}

        # Set defaults and override with YAML values
        self.max_queue_size = config.get("max_queue_size", 60)
        self.target_fps = config.get("target_fps", 3)
        self.gop_size = config.get("gop_size", 3)
        self.segment_duration = config.get("segment_duration", 1.0)
        self.push_address = config.get("encoder_push_address", "tcp://mediaserver:5556")
        self.pull_address = config.get("encoder_pull_address", "tcp://*:5555")
        self.encoding_settings = config.get("encoding_settings")
        
        # ZeroMQ
        context = zmq.Context()

        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(self.push_address)

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(self.pull_address)

        """
        # Thread-safe queue for batches
        self.batch_queue = queue.Queue()

        # Start the processing worker thread
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        """
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        self.codec = CompressionPipeline(self.encoding_settings)

    def run(self):
        batch = []
        start_time_stamp = None

        while True:
            # Receive and deserialize incoming data
            data = pickle.loads(self.pull_socket.recv())
            time_stamp = data["timestamp"]
            print(time_stamp, flush=True)

            # Initialize the start time stamp for the first batch
            if start_time_stamp is None:
                start_time_stamp = time_stamp

            # Calculate current segment length
            curr_segment_duration = time_stamp - start_time_stamp

            if curr_segment_duration <= self.segment_duration:
                # Add to batch if within the segment
                batch.append(data)
            else:
                # Push the batch to the processing queue
                self.executor.submit(self.process, batch)
                #self.batch_queue.put(batch)

                # Reset batch and adjust start time
                start_time_stamp += self.segment_duration  # Ensure fixed intervals
                batch = [data]

    """
    def worker(self):
        while True:
            # Wait for a batch to process
            batch = self.batch_queue.get()
            if batch is None:
                break  # Exit worker thread if None is received
            self.process(batch)
    """

    def process(self, batch):
        # Sampling
        sampled_batch = self.sample(batch)

        # Compression
        compressed_data = self.compress_batch(sampled_batch)

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
        n = int(self.segment_duration * self.target_fps)  # Total number of frames to sample
        step = self.segment_duration / n

        target_timestamps = [start_time + i * step for i in range(n)]

        sampled_batch = []
        for target in target_timestamps:
            closest_frame = min(batch, key=lambda item: abs(item["timestamp"] - target))
            sampled_batch.append(closest_frame)

        return sampled_batch

    def compress_batch(self, sampled_batch):
        """
        Mock compression function. Currently, just returns the input.
        """
        data = {}
        data["timestamp"] = sampled_batch[0]["timestamp"]
        data["segment_duration"] = self.segment_duration
        data["frame_rate"] = self.target_fps

        compressed_data, sideinfo = self.codec.compress(sampled_batch)

        return compressed_data  

        

    def serialize_data(self, data):
        """
        Serializes data using pickle.
        """
        return pickle.dumps(data)

            
if __name__ == "__main__":
    enc = Encoder(config_file="./shared/config.yaml")
    enc.run()