import time
import asyncio
import websockets
import numpy as np
import json
import trimesh
import signal
import sys

def signal_handler(sig, frame):
    print("Termination signal received. Exiting...")
    asyncio.get_event_loop().stop()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


start_frame = 1480
num_frames = 10
file_template = "test_data/redandblack_vox10_{:04d}.ply"
i = 0

# Preload point cloud data
preloaded_data = []

def preload_point_clouds():
    global preloaded_data
    for file in [file_template.format(i) for i in range(start_frame, start_frame + num_frames)]:
        try:
            # Load the PLY file using trimesh
            point_cloud = trimesh.load(file, process=False)
            points = np.asarray(point_cloud.vertices)
            colors = np.asarray(point_cloud.visual.vertex_colors[:, :3])  # Extract RGB values

            # Serialize to JSON format
            data = {
                "points": points.tolist(),
                "colors": colors.tolist(),
            }
            preloaded_data.append(data)
            print(f"Preloaded data for {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

def get_point_cloud_data(idx):
    return preloaded_data[idx]["points"], preloaded_data[idx]["colors"]


async def send_point_cloud(websocket):
    i = 0 
    while True:
        points, colors = get_point_cloud_data(i)
        i = (i + 1) % (len(preloaded_data))
        
        # Flatten the arrays and combine them into a single byte array
        points_bytes = np.array(points, dtype=np.float32).tobytes()
        colors_bytes = np.array(colors, dtype=np.uint8).tobytes()

        # Send data as a binary message
        data = points_bytes + colors_bytes
        await websocket.send(data)
        
        # Sleep to maintain a steady 30 FPS (1/30 seconds = 0.0333 seconds)
        await asyncio.sleep(1/30)  # 30 frames per second

async def main():
    preload_point_clouds()
    async with websockets.serve(send_point_cloud, "0.0.0.0", 8765):
        await asyncio.Future()  # Keep the server running

if __name__ == "__main__":
    asyncio.run(main())
