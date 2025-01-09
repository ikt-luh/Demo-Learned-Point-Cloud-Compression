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
import concurrent.futures

from codec_single import DecompressionPipeline as DecoderSingle
from codec_parallel import DecompressionPipeline as DecoderParallel

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
        self.decoder_type = config.get("decoder_type")

        # ZeroMQ
        context = zmq.Context()
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(self.decoder_push_address)
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(self.decoder_pull_address)

        # Bounded queue for frame buffering
        self.queue = queue.Queue(maxsize=self.max_queue_size)

        if self.decoder_type == "Single":
            self.codec = DecoderSingle()
        else:
            self.codec = DecoderParallel()
        
         # Thread pool for parallel decoding
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Adjust workers as needed


    def decode_and_send(self, data):
        """
        Decode the received data and send it back.
        """
        codec_info, data = data
        data = pickle.loads(data)

        if codec_info == "unified":
            print(f"{time.time()} Sending to decoder", flush=True)
            data_bitstream = data
            decompressed_batch = self.codec.decompress(data_bitstream)

            # Send decompressed data back via the socket
            self.push_socket.send(pickle.dumps(decompressed_batch))
        else:
            self.push_socket.send(data)

    def run(self):
        """
        Continuously receive, decode, and send data.
        """
        while True:
            # Receive data
            raw_data = self.pull_socket.recv()
            data = pickle.loads(raw_data)

            # Submit decoding task to the thread pool
            self.executor.submit(self.decode_and_send, data)

    

if __name__ == "__main__":
    decoder = Decoder("./shared/config.yaml")
    decoder.run()
