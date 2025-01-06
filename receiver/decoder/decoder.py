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


    def run(self):
        """
        Main loop
        """
        while True:
            data = self.pull_socket.recv()
            data = pickle.loads(data)
            print("Received data", flush=True)

            data_bitstream = data[0]
            decompressed_batch = self.codec.decompress(data_bitstream)

            decompressed_batch = pickle.dumps(decompressed_batch)
            self.push_socket.send(decompressed_batch)
    

if __name__ == "__main__":
    decoder = Decoder("./shared/config.yaml")
    decoder.run()
