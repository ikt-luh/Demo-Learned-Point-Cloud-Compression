import zmq
import pyrealsense2 as rs


class Capturer():
    def __init__(self, camera="realsense"):
        """
        """
        self.camera = camera
        self.decimate = 0

        if self.camera == "realsense":
            self.setup_realsense()
        else:
            pass

        # ZMQ
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        self.socket.connect("tcp://encoder:5555")
        

    def run(self):
        while True:
            # Grab the point cloud
            if self.camera == "realsense":
                pointcloud = self.get_realsense_frames()
            else: 
                pointcloud = self.get_zed_frames()

            # Serialize it
            serialized_data = self.serialize_pointcloud(pointcloud)

            # Send it
            socket.send(serialized_data)

    def setup_realsense(self):
        device_list = rs.device_list()
        print(device_list)
        for dev in device_list:
            print(dev)

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

    def get_zed_frames(self):
        #TODO
        return None

    def serialize_pointcloud(self, data):
        #TODO
        return data


if __name__ == "__main__":
    capturer = Capturer()
    capturer.run()
