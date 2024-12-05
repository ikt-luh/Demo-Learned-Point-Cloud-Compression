import zmq


class Encoder():
    def __init__(self):
        context = zmq.Context()

        # ZeroMQ
        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5555")

        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.bind("tcp://media_server:5556")


    def run(self):
        while True:
            # Receiving
            serialized_data = self.pull_socket.recv()
            pointcloud = self.deserialize_pointcloud(raw_data)

            # Compression
            compressed_data = self.compress(pointcloud)

            # Sending
            self.push_socket.send(compressed_data)


    def compress(self, data):
        return data

    def deserialize_pointcloud(self, data):
        return data


if __name__ == "__main__":
    encoder = Encoder()
    encoder.run()
        
