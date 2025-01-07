import os
import random
import time
import yaml
import threading
import queue
import zmq
import pickle
import numpy as np
import torch

from codec_single import DecompressionPipeline

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

class Decoder:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file) 

        self.queue_cleanup_threshold = 2
        self.max_queue_size = config.get("max_queue_size")
        self.decoder_push_address = config.get("decoder_push_address")
        self.decoder_pull_address = config.get("decoder_pull_address")

        # ZeroMQ
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(self.decoder_push_address)
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(self.decoder_pull_address)

        # Bounded queue for frame buffering
        self.queue = queue.Queue(maxsize=self.max_queue_size)

        self.codec = DecompressionPipeline()

    def fill_queue(self):
        while True:
            data = self.pull_socket.recv()
            data = pickle.loads(data)
            print("{} Received data".format(time.time()), flush=True)

            # Cleanup if queue exceeds threshold
            while self.queue.qsize() > self.queue_cleanup_threshold:
                print("Queue exceeded cleanup threshold. Dropping excess data.", flush=True)
                _ = self.queue.get()

            # Enqueue data
            self.queue.put(data)

    def decode(self):
        """
        Thread to decode data and push it back directly.
        """
        while True:
            if self.queue.qsize() > 0:
                data = self.queue.get(timeout=1)  # Retrieve data from queue
                codec_info, data = data #pickle.loads(data)
                data = pickle.loads(data)

                if codec_info == "unified":
                    data_bitstream = data
                    print("{} Sending to decoder".format(time.time()), flush=True)
                    decompressed_batch = self.codec.decompress(data_bitstream)

                    # Send decompressed data back via the socket
                    self.push_socket.send(pickle.dumps(decompressed_batch))
                else:
                    print(pickle.loads(data))
                    self.push_socket.send(data)
            else:
                time.sleep(0.05)



    def run(self):
        """
        Start all threads.
        """
        fill_thread = threading.Thread(target=self.fill_queue, daemon=True)
        decode_push_thread = threading.Thread(target=self.decode, daemon=True)

        fill_thread.start()
        decode_push_thread.start()

        while True:
            time.sleep(1)

    

if __name__ == "__main__":
    decoder = Decoder("./shared/config.yaml")
    decoder.run()
