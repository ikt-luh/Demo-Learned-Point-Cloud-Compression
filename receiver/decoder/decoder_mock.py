import random
import time
import yaml
import threading
import queue
import zmq
import pickle
import numpy as np
import torch

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


    def run(self):
        """
        Main loop
        """
        while True:
            time.sleep(0.1)
            data = self.pull_socket.recv()
            print("Received data")
            time.sleep(random.uniform(0.5, 0.8))
            self.push_socket.send(data)
    

if __name__ == "__main__":
    print("TESTING")
    decoder = Decoder()
    decoder.run()
