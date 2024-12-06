import time
import yaml
import threading
import queue
import zmq
import pickle
import numpy as np
import torch

from unified.model import model


class Encoder():
    def __init__(self, max_queue_size=60):
        context = zmq.Context()

        # ZeroMQ
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect("tcp://mediaserver:5556")

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5555")

        # Bounded queue for frame buffering
        self.queue = queue.Queue(maxsize=max_queue_size)

        # Start a thread to read from the pull socket and populate the queue
        self.receiver_thread = threading.Thread(target=self.receive_data)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()


        # Torch setup
        self.device = torch.device("cuda")
        torch.no_grad()

        # Model
        config_path = "./unified/configs/CVPR_inverse_scaling.yaml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self.codec = model.ColorModel(config["model"])
        self.codec.update()
        self.codec.to(self.device)



    def receive_data(self):
        while True:
            # Receiving
            serialized_data = self.pull_socket.recv()
            try:
                if self.queue.full():
                    self.queue.get()
                self.queue.put(serialized_data)
            except Exception as e:
                print(f"Error in receiver thread: {e}")


    def run(self):
        while True:
            serialized_data = self.queue.get()

            pointcloud = self.deserialize_data(serialized_data)

            # Compression
            compressed_data = self.compress(pointcloud)

            # Sending
            self.push_socket.send(compressed_data)


    def compress(self, data):
        #return msgpack.packb(data, bytes=True)
        input_pc = torch.tensor(
                np.concatenate((data["points"], data["colors"]), axis=1), 
                device=self.device,
                dtype=torch.float)
        q = torch.tensor([[1.0, 1.0]], device=self.device, dtype=torch.float)
        
        torch.cuda.synchronize()
        t0 = time.time()
        res = self.codec.compress(input_pc, q, block_size=1024)
        bits = res[0][0]
        print(str(( (len(bits[0][0]) + len(bits[1][0])) * 8) / input_pc.shape[0]) + " bpp")
        t1 = time.time()
        print(t1 - t0)

            
        return pickle.dumps(data)

    def deserialize_data(self, data):
        #return msgpack.unpackb(data, raw=False)
        return pickle.loads(data)


if __name__ == "__main__":
    encoder = Encoder()
    encoder.run()
        
