import os
import yaml
import pickle
import zmq
import pyzed.sl as sl
import numpy as np
import time
import open3d as o3d


class Capturer():
    def __init__(self, config_file=None): 
        # Load settings from YAML if a file is provided
        if config_file:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
        else:
            config = {}

        self.camera = config.get("camera")
        self.mode = config.get("mode", "demo") # None, Record, Playback
        self.recording_path = config.get("recording_path", None)
        self.depth_clip = config.get("depth_clip", -1.5)
        self.voxel_size = config.get("voxel_size", 0.01)
        self.max_points = config.get("max_points", None)
        self.capturer_push_address = config.get("capturer_push_address")

        # For Playback mode
        self.frame_id = 0
        self.frame_buffer = None

        # Camera setup
        if self.camera == "zed":
            self.setup_zed()

        # ZMQ
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect(self.capturer_push_address)
        

    def run(self):
        while True:
            # Grab the point cloud
            if self.mode == "playback":
                pointcloud = self.playback_frames()
            else:
                pointcloud = self.get_zed_frames()

            # Serialize it
            if pointcloud is not None:
                serialized_data = self.serialize_pointcloud(pointcloud)
                self.socket.send(serialized_data)


    def setup_zed(self):
        """
        """
        init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA, # ULTRA, QUALITY, PERFORMANCE
                coordinate_units=sl.UNIT.METER,
                coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 15

        self.res = sl.Resolution()
        self.res.width = 640
        self.res.height = 480

        self.zed = sl.Camera()
        status = self.zed.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        camera_model = self.zed.get_camera_information().camera_model

        self.pointcloud = sl.Mat(self.res.width, self.res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    def get_zed_frames(self):
        """
        Grab a frame from a ZED Camera and computes the Point Cloud
        """
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            timestamp = time.time()
            self.zed.retrieve_measure(self.pointcloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, self.res)

            # Get Point Cloud from frame
            data = self.pointcloud.numpy()
            points = data[:,:,:3].reshape(-1, 3)

            # Color processing
            float_colors = data[:,:,3].reshape(-1, 1)
            int_colors = float_colors.view(np.uint32)
            colors = np.stack([((int_colors >> (8 * i)) & 0xFF) for i in range(3)], axis=-1).reshape(-1, 3)

            # Remove invalid points (e.g., NaN or extreme values)
            valid_mask = np.isfinite(points).all(axis=1) & (points[:, 2] >= self.depth_clip)
            points = points[valid_mask]
            colors = colors[valid_mask] 

            # Voxelize
            o3d_pointcloud = o3d.geometry.PointCloud()
            o3d_pointcloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))  # Open3D uses float64 for points
            o3d_pointcloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)/255.0)  # Colors need to be in range [0, 1]

            # Downsample the point cloud using voxel size
            downsampled_pointcloud = o3d_pointcloud.voxel_down_sample(self.voxel_size)
            points = np.asarray(downsampled_pointcloud.points)
            colors = np.asarray(downsampled_pointcloud.colors)
            points = np.round(points / (self.voxel_size)).astype(np.int16)

            # Remove duplicates
            _, unique_indices = np.unique(points, axis=0, return_index=True)
            points = points[unique_indices]
            colors = colors[unique_indices]

            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)

            if self.max_points is not None and points.shape[0] > self.max_points:
                indices = np.argpartition(points[:, 2], -self.max_points)[-self.max_points:] 
                points = points[indices]
                colors = colors[indices]

            data = { "points": points, "colors": colors, "timestamp": timestamp }


            if self.mode == "record":
                self.record_frame(data)
            return data
        else:
            return None

    def record_frame(self, data):
        if data is None:
            return

        timestamp = data["timestamp"]
        file_path = os.path.join(self.recording_path, f"frame_{self.frame_id:05d}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(data, file)

        self.frame_id += 1


    def playback_frames(self):
        if self.frame_buffer is None:
            files = sorted([f for f in os.listdir(self.recording_path) if f.endswith(".pkl")])

            # Load all frames
            self.frame_buffer = []
            for file_name in files:
                file_path = os.path.join(self.recording_path, file_name)
                with open(file_path, "rb") as file:
                    frame_data = pickle.load(file)
                    self.frame_buffer.append(frame_data)
            
            # Manipulate the time stamps
            start_time = time.time() + 2.0
            offset = self.frame_buffer[0]["timestamp"] 
            for frame in self.frame_buffer:
                frame["timestamp"] = frame["timestamp"] - offset + start_time            

        # Now we can send frames
        if len(self.frame_buffer) > 0:
            data = self.frame_buffer.pop(0)
            play_time = data["timestamp"]

            sleep_time = max(0, play_time - time.time())
            time.sleep(sleep_time)
            return data


                 

    def serialize_pointcloud(self, data):
        return pickle.dumps(data)


if __name__ == "__main__":
    capturer = Capturer("./shared/config.yaml")
    capturer.run()
