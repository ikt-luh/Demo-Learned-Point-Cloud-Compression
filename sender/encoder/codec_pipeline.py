import os
import yaml
import time
import torch
import numpy as np
import MinkowskiEngine as ME
import pickle
import threading
import concurrent.futures
from bitstream import BitStream

import shared.utils as utils
from shared.notifying_queue import NotifyingQueue
from unified.model import model

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


class CompressionPipeline:
    def __init__(self, settings):
        # Define GPU device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model
        base_path = "./unified/results/"
        self.compression_model = self.load_model(base_path)
        self.settings = settings

        # Queue's
        self.analysis_queue = NotifyingQueue()
        self.hyper_analysis_queue = NotifyingQueue()
        self.geometry_compression_queue = NotifyingQueue()
        self.factorized_model_queue = NotifyingQueue()
        self.hyper_synthesis_queue = NotifyingQueue()
        self.adaptive_compression_queue_y = NotifyingQueue()
        self.adaptive_compression_queue_points = NotifyingQueue()
        self.adaptive_compression_queue_priors = NotifyingQueue()
        self.results_queue = NotifyingQueue()

        # Start threads
        self.threads = []
        self.threads.append(threading.Thread(target=self.run_analysis, daemon=True))
        self.threads.append(threading.Thread(target=self.run_geometry_compression, daemon=True))
        self.threads.append(threading.Thread(target=self.run_hyper_analysis, daemon=True))
        self.threads.append(threading.Thread(target=self.run_factorized_bottleneck, daemon=True))
        self.threads.append(threading.Thread(target=self.run_hyper_synthesis, daemon=True))
        self.threads.append(threading.Thread(target=self.run_adaptive_compression, daemon=True))

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.settings))

        for thread in self.threads:
            thread.start()
        
    def load_model(self, base_path):
        model_name = "demo_small"
        #model_name = "model_inverse_nn"
        config_path = os.path.join(base_path, model_name, "config.yaml")
        weights_path = os.path.join(base_path, model_name, "weights.pt")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        compression_model = model.ColorModel(config["model"])
        compression_model.load_state_dict(torch.load(weights_path, map_location=self.device))

        compression_model.to(self.device)
        #compression_model.entropy_model.gaussian_conditional.to(torch.device("cpu"))
        #compression_model.entropy_model.scale_nn.to(torch.device("cpu"))
        compression_model.update()
        compression_model.eval()

        return compression_model

    def run_analysis(self):
        while True:
            pointclouds = self.analysis_queue.get()

            # Step 1 (Analysis)
            with torch.no_grad():
                y, k, y_points, t_1 = self.analysis_step(pointclouds) 

            result = {
                "y": y,
                "k" : k,
                "y_points": y_points,
                "t_1" : t_1,
            }

            self.hyper_analysis_queue.put(result)
            self.geometry_compression_queue.put(result)
            self.adaptive_compression_queue_y.put(result)

    def run_geometry_compression(self):
        while True:
            result = self.geometry_compression_queue.get()
            y_points = result["y_points"]

            # Step 6 (Geometry Compression with G-PCC)
            points_streams, t_6 = self.geometry_compression_step(y_points)

            result["points_streams"] = points_streams
            result["t_6"] = t_6
            self.adaptive_compression_queue_points.put(result)

    def run_hyper_analysis(self):
        while True:
            result = self.hyper_analysis_queue.get()
            y = result["y"]

            # Step 2 (Hyper Analysis)
            z, t_2 = self.hyper_analysis_step(y)

            result["z"] = z
            result["t_2"] = t_2
            self.factorized_model_queue.put(result)


    def run_factorized_bottleneck(self):
        while True:
            result = self.factorized_model_queue.get()
            z = result["z"]

            # Step 3 (Factorized Entropy Model)
            z_hat, z_strings, z_shapes, z_points, t_3 = self.factorized_model_step(z)


            result["z_hat"] = z_hat
            result["z_strings"] = z_strings
            result["z_shapes"] = z_shapes
            result["z_points"] = z_points
            result["t_3"] = t_3
            self.hyper_synthesis_queue.put(result)

    def run_hyper_synthesis(self):
        while True:
            result = self.hyper_synthesis_queue.get()
            z_hat = result["z_hat"]

            # Step 4 (Hyper Synthesis)
            gaussian_params, t_4 = self.hyper_synthesis_step(z_hat)

            result["gaussian_params"] = gaussian_params
            result["t_4"] = t_4
            self.adaptive_compression_queue_priors.put(result)

    def run_adaptive_compression(self):
        while True:
            result_y = self.adaptive_compression_queue_y.get()
            result_points = self.adaptive_compression_queue_points.get()
            result_priors = self.adaptive_compression_queue_priors.get()

            # Merge the dicts
            result = {**result_y, **result_points, **result_priors}


            # Submit tasks to the thread pool
            futures = []
            for i, setting in enumerate(self.settings):
                futures.append(self.thread_pool.submit(self.process_setting, result, i, setting))

            # Collect results
            compressed_data = {}
            t_5s, t_7s = [], []
            for future in concurrent.futures.as_completed(futures):
                print("Collecting")
                i, byte_array, t_5, t_7 = future.result()
                compressed_data[i] = byte_array
                t_5s.append(t_5)
                t_7s.append(t_7)

            result["compressed_data"] = compressed_data
            result["t_5s"] = t_5s
            result["t_7s"] = t_7s
            self.results_queue.put(result)
            # Define the task for parallel execution

    def process_setting(self, result, i, setting):
        y = result["y"]
        y_points = result["y_points"]
        gaussian_params = result["gaussian_params"]
        z_strings = result["z_strings"]
        z_shapes = result["z_shapes"]
        points_streams = result["points_streams"]
        k = result["k"]
        q = torch.tensor([setting], dtype=torch.float, device=self.device)

        # Step 5 (Gaussian Entropy Model)
        y_strings, y_shapes, t_5 = self.gaussian_model_step(y, y_points, q, gaussian_params)

        # Step 7 (Bitstream Writing)
        byte_array, t_7 = self.make_bitstream(y_strings, z_strings, y_shapes, z_shapes, points_streams, k, q)

        return i, byte_array, t_5, t_7

    def compress(self, data):
        """
        Main compression pipeline method.
        Runs all steps from analysis to bitstream writing.
        Supports multiple calls without conflict.
        """
        t_start = time.time()
        # Hanlde Uncompressed Data
        compressed_data = {}
        compressed_data[0] = pickle.dumps(data)

        pointclouds, sideinfo = self.unpack_batch(data)

        self.analysis_queue.put(pointclouds)
        

        result = self.results_queue.get()
        t_1 = result["t_1"]
        t_2 = result["t_2"]
        t_3 = result["t_3"]
        t_4 = result["t_4"]
        t_5s = result["t_5s"]
        t_6 = result["t_6"]
        t_7s = result["t_7s"]
        z_shapes = result["z_shapes"]

        # Append Compressed data
        for key, item in result["compressed_data"].items():
            print(key+1)
            compressed_data[key+1] = item

        # Logging
        num_points = pointclouds.C.shape[0]
        num_frames = len(z_shapes)
        t_sum = t_1 + t_2 + t_3 + t_4 + sum(t_5s) + t_6 + sum(t_7s)
        print("+++++++++++++++++++++++++++++++++++++++++++++++", flush=True)
        print("Step 1 (Analysis): \t\t {:.3f} seconds".format(t_1), flush=True)
        print("Step 2 (Hyper Analysis): \t {:.3f} seconds".format(t_2), flush=True)
        print("Step 3 (Fact. Entr. Model): \t {:.3f} seconds".format(t_3), flush=True)
        print("Step 4 (Hyper Synthesis): \t {:.3f} seconds".format(t_4), flush=True)
        print("Step 5 (G-PCC): \t\t\t {:.3f} seconds".format(t_6), flush=True)
        for i, (t_5, t_7) in enumerate(zip(t_5s, t_7s)):
            print("\tQuality {}".format(i), flush=True)
            print("\tStep 6 (Gaussian Model): \t {:.3f} seconds".format(t_5), flush=True)
            print("\tStep 7 (Bitstream Writer): \t {:.3f} seconds".format(t_7), flush=True)
            print("Bitstream length: \t\t {} bits".format(len(compressed_data[i+1]) * 8), flush=True)
            print("BPP: \t\t\t\t {:.3f}".format((len(compressed_data[i+1]) * 8) / num_points), flush=True)
        print("-----------------------------------------------", flush=True)
        print("Num Points - (Total): {} Per Frame {:.3f}".format(num_points, num_points/num_frames), flush=True)
        print("Num Frames: {} ".format(num_frames), flush=True)
        print("Compression time: {:.3f} sec".format(t_sum))
        print("+++++++++++++++++++++++++++++++++++++++++++++++", flush=True)

        t_end = time.time()
        print("Compression time: {:.3f} sec".format(t_end - t_start), flush=True)
        return compressed_data, sideinfo 


    def unpack_batch(self, batch):
        """
        Unpacks the batch to receive torch point cloud and a list of side-info
        """
        sideinfo = []

        points, colors = [], []
        for item in batch:
            # Sometimes we miss a frame? 
            if not "points" in item.keys():
                continue

            item_points = item.pop("points")
            item_colors = item.pop("colors")
            sideinfo.append(item)
            points.append(item_points)
            colors.append(item_colors)

        # From list to torch tensors in the expected shape
        points, colors = utils.stack_tensors(points, colors)
        colors = torch.cat([torch.ones((colors.shape[0], 1)), colors], dim=1)
        points = points.to(self.device, dtype=torch.float)
        colors = colors.to(self.device, dtype=torch.float)

        pointcloud = ME.SparseTensor(
            coordinates=points,
            features=colors,
            device=self.device
        )
        return pointcloud, sideinfo


    def analysis_step(self, data):
        """ Step 1: Analysis (GPU) """
        t0 = time.time()
        y, k = self.compression_model.g_a(data)
        torch.cuda.synchronize()

        y = utils.sort_tensor(y)
        y_points = utils.get_points_per_batch(y, batch_dim=False)

        t1 = time.time()
        t_step = t1 - t0
        return y, k, y_points, t_step


    def hyper_analysis_step(self, y):
        """ Step 2: Hyper Analysis """
        t0 = time.time()
        z = self.compression_model.entropy_model.h_a(y)
        torch.cuda.synchronize()
        t1 = time.time()
        t_step = t1 - t0
        return z, t_step


    def factorized_model_step(self, z):
        """ Step 3: Factorized Entropy Model """
        t0 = time.time()

        z = utils.sort_tensor(z)

        z_feats = utils.get_features_per_batch(z)
        z_shapes = [[feat.shape[0]] for feat in z_feats]
        z_feats = [feat.t().unsqueeze(0) for feat in z_feats]
        
        z_strings, z_hats = [], []
        for z_feat, shape in zip(z_feats, z_shapes):
            z_string = self.compression_model.entropy_model.entropy_bottleneck.compress(z_feat)
            z_strings.append(z_string)
            z_hat = self.compression_model.entropy_model.entropy_bottleneck.decompress(z_string, shape)
            z_hats.append(z_hat[0].t())

        z_points = utils.get_points_per_batch(z)

        z_hat = ME.SparseTensor(
            coordinates=z.C.clone(),
            features=torch.cat(z_hats, dim=0),
            tensor_stride=32,
            device=self.device
        )

        t1 = time.time()
        t_step = t1 - t0
        return z_hat, z_strings, z_shapes, z_points, t_step


    def hyper_synthesis_step(self, z_hat):
        t0 = time.time()

        gaussian_params = self.compression_model.entropy_model.h_s(z_hat)

        torch.cuda.synchronize()
        t1 = time.time()
        t_step = t1 - t0
        return gaussian_params, t_step


    def gaussian_model_step(self, y, y_points, q, gaussian_params):
        t0 = time.time()
        
        """
        y = ME.SparseTensor(
            coordinates=y.C,
            features=y.F,
            tensor_stride=8,
            device=torch.device("cpu")
        )
        gaussian_params = ME.SparseTensor(
            coordinates=gaussian_params.C,
            features=gaussian_params.F,
            tensor_stride=8,
            device=torch.device("cpu")
        )
        q = q.cpu()
        """

        t_start = time.time()
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())

        gaussian_params = utils.get_features_per_batch(gaussian_params_feats, y.C)
        y_feats = utils.get_features_per_batch(y)
        t_end = time.time()
        print("Processing {}".format(t_end - t_start))

        # Scale NN
        t_start = time.time()
        scale = self.compression_model.entropy_model.scale_nn(q) + self.compression_model.entropy_model.eps
        scale = scale[:].t().unsqueeze(0)#.cpu()
        t_end = time.time()

        print("NN {}".format(t_end - t_start))
        y_strings, shapes = [], []
        for y_feat, gaussian_param in zip(y_feats, gaussian_params):
            scales_hat, means_hat = gaussian_param.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            t_start = time.time()
            # Alternative Approach to building indexes
            scales_hat = self.compression_model.entropy_model.gaussian_conditional.lower_bound_scale(scales_hat * scale)
            scale_table_tensor = torch.tensor(self.compression_model.entropy_model.gaussian_conditional.scale_table[:-1], device=scales_hat.device)
            indexes = torch.sum(scales_hat.unsqueeze(-1) > scale_table_tensor, dim=-1).int()

            #indexes = self.compression_model.entropy_model.gaussian_conditional.build_indexes(scales_hat * scale)
            print("Indexes: {}".format(time.time() - t_start))
            t_start = time.time()
            y_string = self.compression_model.entropy_model.gaussian_conditional.compress(
                y_feat.t().unsqueeze(0) * scale,
                indexes,
                means=means_hat * scale
            )
            t_end = time.time()
            print("Compresds: {}".format(t_end - t_start))

            y_strings.append(y_string)
            shapes.append(y_feat.shape[0])

        t1 = time.time()
        t_step = t1 - t0

        return y_strings, shapes, t_step


    def geometry_compression_step(self, y_points):
        t0 = time.time()

        base_directory = "/tmp/"

        timestamp = int(time.time() * 1e6)
        # TODO: Create unique subdirectory names for threads
        # thread_id = threading.get_ident()
        # tmp_dir = os.path.join(base_directory, f"tmp_{thread_id}_{timestamp}_points_enc.ply")
        # bin_dir = os.path.join(base_directory, f"tmp_{thread_id}_{timestamp}_points_bin.ply")

        tmp_dir = os.path.join(base_directory, f"tmp_{timestamp}_points_enc.ply")
        bin_dir = os.path.join(base_directory, f"tmp_{timestamp}_points_enc.bin")

        point_bitstreams = []
        for y_point in y_points:
            point_bitstream = utils.gpcc_encode(y_point, tmp_dir, bin_dir)
            point_bitstreams.append(point_bitstream)

        t1 = time.time()
        t_step = t1 - t0
        return point_bitstreams, t_step

    def make_bitstream(self, y_strings, z_strings, y_shapes, z_shapes, points_streams, ks, q):
        """
        Write the bitstream

        Structure:
        [ num_frames (32) | q_g (32) | q_g (32) | [frame1] ... frame[N] ]

        Frame:  [ Header (128 bits) | Content (whatever) ]
            Header: [y_shape (32) | z_shape (32) | len_points (32) | len_y (32) | len_z (32) | k1,k2,k3 (96)] 
            Content: [ points | z_string | y_string ] 
        """
        t0 = time.time()

        stream = BitStream()

        num_frames = len(y_strings)
        stream.write(num_frames, np.int32)
        stream.write(q[0, 0].cpu(), np.float64)
        stream.write(q[0, 1].cpu(), np.float64)

        """
        print("-------------")
        print("Header:")
        print("Num Frames: {}".format(num_frames))
        print("Q: {} {}".format(q[0,0], q[0,1]))
        print("-------------")
        """

        for i in range(num_frames):
            points = points_streams[i]
            y_string, z_string = y_strings[i], z_strings[i]
            y_shape, z_shape = y_shapes[i], z_shapes[i]
            q = q

            # Header
            stream.write(y_shape, np.int32)
            stream.write(z_shape, np.int32)
            stream.write(len(points), np.int32)
            stream.write(len(y_string[0]), np.int32)
            stream.write(len(z_string[0]), np.int32)

            stream.write(ks[0][i], np.int32)
            stream.write(ks[1][i], np.int32)
            stream.write(ks[2][i], np.int32)

            # Content
            stream.write(points, bytes)
            stream.write(y_string[0], bytes)
            stream.write(z_string[0], bytes)

            """
            print("Frame {}:".format(i))
            print("y_shape: \t{}".format(y_shape))
            print("z_shape: \t{}".format(z_shape))
            print("points_len: \t{}".format(len(points)))
            print("y_len: \t\t{}".format(len(y_string[0])))
            print("z_len: \t\t{}".format(len(z_string[0])))
            print("k1: \t{}".format(ks[0][i]))
            print("k2: \t{}".format(ks[1][i]))
            print("k3: \t{}".format(ks[2][i]))
            print("-----------")
            """

        bit_string = stream.__str__()
        byte_array = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

        t1 = time.time()
        t_step = t1 - t0
        return byte_array, t_step