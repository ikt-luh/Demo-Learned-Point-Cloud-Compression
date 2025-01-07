from queue import Queue
import pickle
import numpy as np

class BufferManager:
    def __init__(self, max_buffer_size, target_fps):
        self.playout_buffer = Queue(maxsize=max_buffer_size)
        self.target_fps = target_fps

    def prepare_for_rendering(self, decoded_data):
        data = pickle.loads(decoded_data)
        for frame in data:
            points = frame["points"] + 100
            colors = 255 * frame["colors"]

            points_bytes = np.array(points, dtype=np.float32).tobytes()
            colors_bytes = np.array(colors, dtype=np.uint8).tobytes()
            packed_data = points_bytes + colors_bytes

            self.playout_buffer.put(packed_data)
