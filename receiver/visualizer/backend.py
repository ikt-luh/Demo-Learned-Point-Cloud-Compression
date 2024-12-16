import time
import sys
import zmq
import pickle
import websockets
import numpy as np
import threading
import asyncio

class Backend:
    def __init__(self):
        # ZeroMQ
        self.context = zmq.Context()
        self.pull_socket = self.context.socket(zmq.PULL)
        self.pull_socket.bind("tcp://*:5556")  # Bind to receive data

        # WebSocket
        self.websocket = None  # Placeholder for the single client connection

    def zmq_listener(self):
        """Listen for incoming data on ZeroMQ and pass it to the WebSocket."""
        while True:
            data = self.pull_socket.recv()
            data = pickle.loads(data)
            print("Data received {}".format(time.time()))  # Blocking call to receive data
            print("Timestamp {}".format(time.time() - data[0]["timestamp"]))  # Blocking call to receive data
            sys.stdout.flush()
            self.pack_and_send(data)

    async def websocket_handler(self, websocket, path):
        """Handle WebSocket communication (single client)."""
        self.websocket = websocket
        try:
            for message in websocket:  # Handle incoming messages if needed
                print(f"Client message: {message}")
        except websockets.ConnectionClosed:
            print("WebSocket client disconnected")
        finally:
            self.websocket = None

    def pack_and_send(self, data):
        """Pack the data and send it to the WebSocket client."""
        if self.websocket is None:
            return  # No active WebSocket client, drop the data

        # Simulate unpacking and processing data
        points = np.random.rand(100, 3)  # Example: generate fake points
        colors = np.random.randint(0, 255, (100, 3))  # Example: generate fake colors

        # Flatten arrays and convert to byte format
        points_bytes = points.astype(np.float32).tobytes()
        colors_bytes = colors.astype(np.uint8).tobytes()
        packed_data = points_bytes + colors_bytes

        # Send data
        self.websocket.send(packed_data)

    async def run(self):
        """Run both ZeroMQ and WebSocket servers."""
        # Start ZeroMQ listener in a separate thread
        zmq_thread = threading.Thread(target=self.zmq_listener, daemon=True)
        zmq_thread.start()

        # Start WebSocket server (blocking)
        start_server = websockets.serve(self.websocket_handler, "0.0.0.0", 8080)

        await start_server
        await asyncio.Future()


if __name__ == "__main__":
    print("Starting backend")
    backend = Backend()
    asyncio.run(backend.run())
