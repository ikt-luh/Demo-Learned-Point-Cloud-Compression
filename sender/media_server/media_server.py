import numpy
import pickle
import time
import zmq

class Server():
    def __init__(self):
        context = zmq.Context()

        # ZeroMQ
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5556")

    def run(self):
        while True:
            serialized_data = self.pull_socket.recv()
            data = self.deserialize_data(serialized_data)
            print("Got some data")
            print(time.time() - data["timestamp"])
            print(data["points"].shape)

    def deserialize_data(self, data):
        #return msgpack.unpackb(data, raw=False)
        return pickle.loads(data)

if __name__ == "__main__":
    server = Server()
    server.run()
