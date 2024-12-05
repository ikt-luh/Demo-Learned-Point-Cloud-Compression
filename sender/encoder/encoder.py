import threading
import queue
import zmq
import pickle
import numpy


class Encoder():
    def __init__(self, max_queue_size=30):
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
        return pickle.dumps(data)

    def deserialize_data(self, data):
        #return msgpack.unpackb(data, raw=False)
        return pickle.loads(data)


if __name__ == "__main__":
    encoder = Encoder()
    encoder.run()
        
