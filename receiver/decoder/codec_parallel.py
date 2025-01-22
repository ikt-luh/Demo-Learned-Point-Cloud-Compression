import os
import yaml
import time
import torch
import numpy as np
import MinkowskiEngine as ME
from bitstream import BitStream
import threading

import shared.utils as utils
from shared.notifying_queue import NotifyingQueue
from unified.model import model

LOG = True

torch.set_grad_enabled(False)

class DecompressionPipeline:
    def __init__(self):
        # Define GPU device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model
        base_path = "./unified/results/"
        self.decompression_model = self.load_model(base_path)
        
        # Queue's
        self.geometry_hyper_queue = NotifyingQueue()
        self.hyper_synthesis_queue = NotifyingQueue()
        self.gaussian_queue = NotifyingQueue()
        self.synthesis_queue = NotifyingQueue()
        self.results_queue = NotifyingQueue()

        # Start threads
        self.threads = []
        self.threads.append(threading.Thread(target=self.run_geometry_hyper, daemon=True))
        self.threads.append(threading.Thread(target=self.run_hyper_synthesis, daemon=True))
        self.threads.append(threading.Thread(target=self.run_gaussian_bottleneck, daemon=True))
        self.threads.append(threading.Thread(target=self.run_synthesis, daemon=True))

        for thread in self.threads:
            thread.start()

        print("Pipeline Initialized (Parallel)", flush=True)


    def load_model(self, base_path):
        #model_name = "model_inverse_nn"
        model_name = "demo_small"
        config_path = os.path.join(base_path, model_name, "config.yaml")
        weights_path = os.path.join(base_path, model_name, "weights.pt")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        decompression_model = model.ColorModel(config["model"])
        decompression_model.load_state_dict(torch.load(weights_path, map_location=self.device))

        decompression_model.to(self.device)
        decompression_model.update()
        decompression_model.eval()

        return decompression_model

    def run_geometry_hyper(self):
        while True:
            compressed_data = self.geometry_hyper_queue.get()

            # Step 1 (Bitstream Reading)
            y_strings, z_strings, y_shapes, z_shapes, points_streams, ks, q, t_1 = self.read_bitstream_batched(compressed_data)

            # Step 2 (Geometry Decompression)
            y_points, t_2 = self.geometry_decompression_step(points_streams)

            # Step 3 (Factorized Entropy Model)
            z_hats, t_3 = self.factorized_model_step_batched(z_strings, z_shapes, y_points)

            result = {
                "y_strings" : y_strings,
                "z_strings" : z_strings,
                "y_shapes" : y_shapes,
                "z_shapes" : z_shapes,
                "points_streams" : points_streams,
                "ks" : ks,
                "q" : q,
                "y_points" : y_points,
                "z_hats" : z_hats,
                "t_1" : t_1, 
                "t_2" : t_2, 
                "t_3" : t_3, 
            }
            self.hyper_synthesis_queue.put(result)

    def run_hyper_synthesis(self):
        while True:
            result = self.hyper_synthesis_queue.get()
            z_hats = result["z_hats"]

            # Step 4 (Hyper Synthesis)
            with torch.no_grad():
                gaussian_params, t_4 = self.hyper_synthesis_step(z_hats)

            result["gaussian_params"] = gaussian_params
            result["t_4"] = t_4
            self.gaussian_queue.put(result)


    def run_gaussian_bottleneck(self):
        while True:
            result = self.gaussian_queue.get()
            y_strings = result["y_strings"]
            y_shapes = result["y_shapes"]
            y_points = result["y_points"]
            q = result["q"]
            gaussian_params = result["gaussian_params"]

            # Step 5 (Gaussian Entropy Model)
            y_hat, t_5 = self.gaussian_model_step_batched(y_strings, y_shapes, y_points, q, gaussian_params)
            #y_hat, t_5 = self.gaussian_model_step(y_strings, y_shapes, y_points, q, gaussian_params)

            result["y_hat"] = y_hat
            result["t_5"] = t_5

            self.synthesis_queue.put(result)

    def run_synthesis(self):
        while True:
            result = self.synthesis_queue.get()
            y_hat = result["y_hat"]
            ks = result["ks"]

            # Step 6 (Hyper Synthesis)
            with torch.no_grad():
                reconstructed_pointcloud, t_6 = self.hyper_synthesis(y_hat, ks)

            result["reconstructed_pointcloud"] = reconstructed_pointcloud
            result["t_6"] = t_6

            self.results_queue.put(result)

    def decompress(self, compressed_data):
        """
        Main decompression pipeline method.
        Runs all steps from bitstream reading to reconstruction.
        """
        t_start = time.time()

        self.geometry_hyper_queue.put(compressed_data)

        result = self.results_queue.get()

        reconstructed_pointcloud = result["reconstructed_pointcloud"]

        # Postprocessing: Pack data back into batches
        final_data, t_7 = self.pack_batches(reconstructed_pointcloud)

        sideinfo = {"time_measurements": {}, "timestamps": {}}
        sideinfo["time_measurements"]["bitstream_reading"] = result["t_1"]
        sideinfo["time_measurements"]["geometry_decompression"] = result["t_2"]
        sideinfo["time_measurements"]["factorized_model"] = result["t_3"]
        sideinfo["time_measurements"]["hyper_synthesis"] = result["t_4"]
        sideinfo["time_measurements"]["guassian_model"] = result["t_5"]
        sideinfo["time_measurements"]["synthesis_transform"] = result["t_6"]

        t_end = time.time()
        print("Decompression time: {:.3f} sec".format(t_end - t_start), flush=True)
        sideinfo["timestamps"]["codec_start"] = t_start
        sideinfo["timestamps"]["codec_end"] = t_end
        print("done")

        return final_data, sideinfo

    def read_bitstream_batched(self, compressed_data):
        """ Step 1: Read the bitstream. """
        t0 = time.time()
        
        stream = BitStream()
        stream.write(compressed_data, bytes)

        y_strings, z_strings = [], []
        y_shapes, z_shapes = [], []
        points_streams, ks = [], [[],[],[]]

        # Header
        num_frames = stream.read(np.int32)
        q_g = stream.read(np.float64)
        q_a = stream.read(np.float64)
        q = torch.tensor([[q_g, q_a]], dtype=torch.float, device=self.device)  # Example

        y_shapes = stream.read(np.int32)
        z_shapes = stream.read(np.int32)

        y_stream_length = stream.read(np.int32)
        z_stream_length = stream.read(np.int32)

        y_stream = stream.read(bytes, int(y_stream_length))
        z_stream = stream.read(bytes, int(z_stream_length))
        y_strings = [y_stream]
        z_strings = [z_stream]

        # Frames
        for i in range(num_frames):
            # Frame Header
            point_stream_length = stream.read(np.int32)

            #ks[0].append(int(stream.read(np.int32) * 1.2))
            ks[0].append(stream.read(np.int32))
            ks[1].append(stream.read(np.int32))
            ks[2].append(stream.read(np.int32))
            
            # Content
            points = stream.read(bytes, int(point_stream_length))
            points_streams.append(points)

        t1 = time.time()
        return y_strings, z_strings, y_shapes, z_shapes, points_streams, ks, q, t1 - t0

    def read_bitstream(self, compressed_data):
        """ Step 1: Read the bitstream. """
        t0 = time.time()
        
        stream = BitStream()
        stream.write(compressed_data, bytes)

        y_strings, z_strings = [], []
        y_shapes, z_shapes = [], []
        points_streams, ks = [], [[],[],[]]

        # Header
        num_frames = stream.read(np.int32)
        q_g = stream.read(np.float64)
        q_a = stream.read(np.float64)
        q = torch.tensor([[q_g, q_a]], dtype=torch.float, device=self.device)  # Example

        # Frames
        for i in range(num_frames):
            # Frame Header
            y_shape = stream.read(np.int32)
            z_shape = stream.read(np.int32)
            point_stream_length = stream.read(np.int32)
            y_stream_length = stream.read(np.int32)
            z_stream_length = stream.read(np.int32)

            #ks[0].append(int(stream.read(np.int32) * 1.2))
            ks[0].append(stream.read(np.int32))
            ks[1].append(stream.read(np.int32))
            ks[2].append(stream.read(np.int32))
            
            # Content
            points = stream.read(bytes, int(point_stream_length))

            y_stream = stream.read(bytes, int(y_stream_length))

            z_stream = stream.read(bytes, int(z_stream_length))
            
            # Sort into datastructures
            y_shapes.append(y_shape)
            z_shapes.append(z_shape)
            y_strings.append(y_stream)
            z_strings.append(z_stream)
            points_streams.append(points)

        t1 = time.time()
        return y_strings, z_strings, y_shapes, z_shapes, points_streams, ks, q, t1 - t0

    def geometry_decompression_step(self, points_streams):
        """ Step 2: Geometry Decompression """
        t0 = time.time()
        
        base_directory = "/tmp/"
        os.makedirs(base_directory, exist_ok=True)

        timestamp = int(time.time() * 1e6)
        # TODO: Create unique subdirectory names for threads
        # thread_id = threading.get_ident()
        # tmp_dir = os.path.join(base_directory, f"tmp_{thread_id}_{timestamp}_points_enc.ply")
        # bin_dir = os.path.join(base_directory, f"tmp_{thread_id}_{timestamp}_points_bin.ply")

        tmp_dir = os.path.join(base_directory, f"tmp_{timestamp}_points_enc.ply")
        bin_dir = os.path.join(base_directory, f"tmp_{timestamp}_points_enc.bin")

        y_points = []
        for points_stream in points_streams:
            y_point = utils.gpcc_decode(points_stream, tmp_dir, bin_dir)
            y_points.append(y_point)

        y_points = utils.stack_tensors(y_points)
        t1 = time.time()
        return y_points, t1 - t0

    def factorized_model_step_batched(self, z_strings, z_shapes, y_points):
        """ Step 3: Factorized Entropy Model """
        t0 = time.time()

        # Get latent coordinates
        latent_coordinates = ME.SparseTensor(
            coordinates=y_points.clone(), 
            features=torch.ones((y_points.shape[0], 1)), 
            tensor_stride=8,
            device=self.device
        )
        latent_coordinates = self.decompression_model.g_s.down_conv(latent_coordinates)
        latent_coordinates = self.decompression_model.g_s.down_conv(latent_coordinates)

        z_points = utils.sort_points(latent_coordinates.C)

        z_hat_feat = self.decompression_model.entropy_model.entropy_bottleneck.decompress(z_strings, [z_shapes])

        z_hat = ME.SparseTensor(
            coordinates=z_points,
            features=z_hat_feat[0].t(),
            tensor_stride=32,
            device=self.device
        )

        t1 = time.time()
        t_step = t1 - t0
        return z_hat, t_step


    def factorized_model_step(self, z_strings, z_shapes, y_points):
        """ Step 3: Factorized Entropy Model """
        t0 = time.time()

        # Get latent coordinates
        latent_coordinates = ME.SparseTensor(
            coordinates=y_points.clone(), 
            features=torch.ones((y_points.shape[0], 1)), 
            tensor_stride=8,
            device=self.device
        )
        latent_coordinates = self.decompression_model.g_s.down_conv(latent_coordinates)
        latent_coordinates = self.decompression_model.g_s.down_conv(latent_coordinates)

        z_points = utils.sort_points(latent_coordinates.C)
        z_points = utils.get_points_per_batch(z_points)

        z_hats = []
        for z_point, z_string, z_shape in zip(z_points, z_strings, z_shapes):
            z_hat_feat = self.decompression_model.entropy_model.entropy_bottleneck.decompress([z_string], [z_shape])
            z_hat = ME.SparseTensor(
                coordinates=z_point,
                features=z_hat_feat[0].t(),
                tensor_stride=32,
                device=self.device
            )
            z_hats.append(z_hat)

        t1 = time.time()
        t_step = t1 - t0
        return z_hats, t_step

    def hyper_synthesis_step(self, z_hat):
        """ Step 4: Hyper Synthesis """
        t0 = time.time()
        #z_hat = utils.batch_sparse_tensors(z_hat, tensor_stride=32)

        """
        # Transfer z_hat to CPU
        z_hat_cpu = ME.SparseTensor(
            coordinates=z_hat.C,
            features=z_hat.F,
            device=torch.device("cpu"),
            tensor_stride=32
        )
        self.decompression_model.entropy_model.h_s.to('cpu')
        gaussian_params = self.decompression_model.entropy_model.h_s(z_hat_cpu)
        gaussian_params = ME.SparseTensor(
            coordinates=gaussian_params.C,
            features=gaussian_params.F,
            device=self.device,
            tensor_stride=8
        )
        """

        gaussian_params = self.decompression_model.entropy_model.h_s(z_hat)

        t1 = time.time()
        t_step = t1 - t0
        return gaussian_params, t1 - t0

    def gaussian_model_step_batched(self, y_strings, y_shapes, y_points, q, gaussian_params):
        """ Step 5: Gaussian Entropy Model """
        t0 = time.time()

        y_points = utils.sort_points(y_points.to(self.device))
        gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())

        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        # Scale NN
        scale = self.decompression_model.entropy_model.scale_nn(q) + self.decompression_model.entropy_model.eps
        scale = scale.unsqueeze(2).repeat(1, 1, scales_hat.shape[-1])
        rescale = torch.tensor(1.0) / scale

        indexes = self.decompression_model.entropy_model.gaussian_conditional.build_indexes(scales_hat * scale)

        q_val = self.decompression_model.entropy_model.gaussian_conditional.decompress(y_strings, indexes)
        q_abs, signs = q_val.abs(), torch.sign(q_val)

        y_q_stdev = self.decompression_model.entropy_model.gaussian_conditional.lower_bound_scale(scales_hat * scale)

        q_offsets = (-1) * self.decompression_model.entropy_model.get_offsets(y_q_stdev,scale)
        q_offsets[q_abs < 0.0001] = (0)

        y_hat_feat = signs * (q_abs + q_offsets)
        y_hat_feat = y_hat_feat * rescale + means_hat

        y_hat = ME.SparseTensor(
            coordinates=y_points,
            features=y_hat_feat[0].t(),
            tensor_stride=8,
            device=self.device
        )

        t1 = time.time()
        return y_hat, t1 - t0    
        
        
    def gaussian_model_step(self, y_strings, y_shapes, y_points, q, gaussian_params):
        """ Step 5: Gaussian Entropy Model """
        t0 = time.time()

        y_points = utils.sort_points(y_points.to(self.device))
        gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())
        gaussian_params = utils.get_features_per_batch(gaussian_params_feats, y_points)

        y_hat_feats = []
        for y_string, y_shape, gaussian_param in zip(y_strings, y_shapes, gaussian_params):
            scales_hat, means_hat = gaussian_param.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            # Scale NN
            scale = self.decompression_model.entropy_model.scale_nn(q) + self.decompression_model.entropy_model.eps
            scale = scale.repeat(means_hat.shape[-1], 1).t().unsqueeze(0)
            rescale = torch.tensor(1.0) / scale

            indexes = self.decompression_model.entropy_model.gaussian_conditional.build_indexes(scales_hat * scale)

            q_val = self.decompression_model.entropy_model.gaussian_conditional.decompress([y_string], indexes)
            q_abs, signs = q_val.abs(), torch.sign(q_val)

            y_q_stdev = self.decompression_model.entropy_model.gaussian_conditional.lower_bound_scale(scales_hat * scale)

            q_offsets = (-1) * self.decompression_model.entropy_model.get_offsets(y_q_stdev, scale)
            q_offsets[q_abs < 0.0001] = (0)

            y_hat_feat = signs * (q_abs + q_offsets)
            y_hat_feat = y_hat_feat * rescale + means_hat
            y_hat_feats.append(y_hat_feat[0].t())

        y_hat = ME.SparseTensor(
            coordinates=y_points,
            features=torch.cat(y_hat_feats, dim=0),
            tensor_stride=8,
            device=self.device
        )

        t1 = time.time()
        return y_hat, t1 - t0

    def hyper_synthesis(self, y_hat, ks):
        """ Step 6: Hyper Synthesis """
        t0 = time.time()

        reconstructed_pointcloud = self.decompression_model.g_s(y_hat, k=ks)

        t1 = time.time()
        return reconstructed_pointcloud, t1 - t0

    def pack_batches(self, pointcloud):
        """ Step 7: Postprocessing and packing. """
        t0 = time.time()

        # Extract coordinates and features from the SparseTensor
        points = pointcloud.C.cpu().numpy()  # Convert to NumPy on CPU
        colors = pointcloud.F.cpu().numpy()  # Convert to NumPy on CPU

        # Initialize the batch list
        num_frames = np.max(points[:, 0]) + 1
        batch = []

        for i in range(num_frames):
            batch_indices = points[:, 0] == i  # Match batch index
            item_points = points[batch_indices][:, 1:]  # Exclude batch index
            item_colors = colors[batch_indices][:]  

            item_colors = np.nan_to_num(item_colors, nan=0.0)  # Replace NaNs with 0

            item_colors = np.clip(item_colors * 255.0, 0, 255) / 255

            data = {
                "points": item_points,  # Shape: (N, D)
                "colors": item_colors,  # Shape: (N, C), scaled to int8
            }
            batch.append(data)

        t1 = time.time()
        return batch, t1 - t0
