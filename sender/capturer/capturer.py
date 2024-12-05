import pickle
import zmq
#import pyrealsense2 as rs
import pyzed.sl as sl
import numpy as np
import time
import open3d as o3d


class Capturer():
    def __init__(self, camera="zed"):
        """
        Constructor
        """
        self.camera = camera
        self.decimate = 0
        self.depth_clip = -1.0
        self.voxel_size = 0.002

        # Camera setup
        if self.camera == "realsense":
            self.setup_realsense()
        elif self.camera == "zed":
            self.setup_zed()
        # ZMQ
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect("tcp://encoder:5555")
        

    def run(self):
        while True:
            # Grab the point cloud
            if self.camera == "realsense":
                pointcloud = self.get_realsense_frames()
            elif self.camera == "zed": 
                pointcloud = self.get_zed_frames()

            # Serialize it
            serialized_data = self.serialize_pointcloud(pointcloud)

            # Send it
            self.socket.send(serialized_data)

    def setup_realsense(self):
        device_list = rs.device_list()

        self.pipe = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipe)
        pipeline_config = config.resolve(pipeline_wrapper)
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.depth, rs.format.bgr8, 30)
        
        self.pipe.start(config)

        # Intrinsics 
        profile = self.pipe.get_active_profile()
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        # Filtering
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.decimate)

        # Point Cloud generation
        self.pointcloud = rs.pointcloud()
        self.colorizer = rs.colorizer()


    def get_realsense_frames(self):
        """
        """
        frames = self.pipe.wait_for_frames()
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = decimate.process(depth_frame)

        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrisics.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Generate pointcloud
        #depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        points = self.pointcloud.calculate(depth_frame)
        self.pointcloud.map_to(color_frame)

        v, t = points.get_vertices(), points.get_texture_coordinates()
        xyz = np.asanyarray(v).view(np.float32).reshape(-1, 3) # XYZ

        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
        colors = color_image[texcoords]

        data = {
                "points": xyz,
                "colors": None
                }
        return data

    def setup_zed(self):
        """
        """
        init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                coordinate_units=sl.UNIT.METER,
                coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
        init.camera_resolution = sl.RESOLUTION.HD1080

        self.res = sl.Resolution()
        self.res.width = 960
        self.res.height = 540

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
            t0 = time.time()
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
            points = np.round(points / self.voxel_size).astype(np.int16)
            print(np.min(points), np.max(points))

            data = { "points": points, "colors": colors, "timestamp": t0 }
        return data

    def serialize_pointcloud(self, data):
        #return msgpack.packb(data, use_bin_type=True)
        return pickle.dumps(data)


if __name__ == "__main__":
    capturer = Capturer()
    capturer.run()
