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

class Decoder:
    def __init__(self, max_queue_size=60, target_fps=3, gop_size=3, segment_duration=2.0):
        context = zmq.Context()

        # ZeroMQ
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect("tcp://client:5555")

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5555")

        # Bounded queue for frame buffering
        self.queue = queue.Queue(maxsize=max_queue_size)

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
    decoder = Decoder()
    decoder.run()
