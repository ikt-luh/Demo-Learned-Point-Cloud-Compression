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
            print("Got some data")

if __name__ == "__main__":
    server = Server()
    server.run()
