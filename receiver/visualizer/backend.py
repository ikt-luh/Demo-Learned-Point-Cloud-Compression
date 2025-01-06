import asyncio
import websockets
import yaml
import zmq
import zmq.asyncio

# Initialize ZeroMQ context and socket (PULL for receiving data)
context = zmq.asyncio.Context()
zmq_socket = context.socket(zmq.PULL)
zmq_socket.bind("tcp://*:5556")  # Connect to the ZeroMQ source

async def handler(websocket):
    print("Client connected")
    try:
        while True:
            # Wait for message
            data = await zmq_socket.recv()
            
            # Send  data to WebSocket client
            await websocket.send(data)  # Assuming the data is a string, adjust as needed

    except websocket.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()  # Keep the server running

# Run the WebSocket server
asyncio.run(main())