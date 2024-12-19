import os
import yaml
import time
import torch
import numpy as np
import MinkowskiEngine as ME
from bitstream import BitStream

import shared.utils as utils
from unified.model import model
class DecompressionPipeline:
    def __init__(self):
        # Define GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        base_path = "./unified/results/"
        self.decompression_model = self.load_model(base_path)

    def load_model(self, base_path):
        model_name = "model_inverse_nn"
        config_path = os.path.join(base_path, model_name, "config.yaml")
        weights_path = os.path.join(base_path, model_name, "weights.pt")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        decompression_model = model.ColorModel(config["model"])
        decompression_model.load_state_dict(torch.load(weights_path))

        decompression_model.to(self.device)
        decompression_model.update()
        decompression_model.eval()

        return decompression_model

    def decompress(self, compressed_data):
        """
        Main decompression pipeline method.
        Runs all steps from bitstream reading to reconstruction.
        """
        t_start = time.time()

        # Step 1 (Bitstream Reading)
        y_strings, z_strings, y_shapes, z_shapes, points_streams, ks, q, t_1 = self.read_bitstream(compressed_data)

        # Step 2 (Geometry Decompression)
        y_points, t_2 = self.geometry_decompression_step(points_streams)

        # Step 3 (Factorized Entropy Model)
        z_hats, t_3 = self.factorized_model_step(z_strings, z_shapes, y_points)

        # Step 4 (Hyper Synthesis)
        gaussian_params, t_4 = self.hyper_synthesis_step(z_hats)

        # Step 5 (Gaussian Entropy Model)
        y_hat, t_5 = self.gaussian_model_step(y_strings, y_shapes, y_points, q, gaussian_params)

        # Step 6 (Hyper Synthesis)
        reconstructed_pointcloud, t_6 = self.hyper_synthesis(y_hat, ks)

        # Postprocessing: Pack data back into batches
        final_data, t_7 = self.pack_batches(reconstructed_pointcloud)

        # Logging
        t_sum = t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7
        print("Step 1 (Bitstream Reading): \t {:.3f} seconds".format(t_1), flush=True)
        print("Step 2 (Geometry Decompression): \t {:.3f} seconds".format(t_2), flush=True)
        print("Step 3 (Factorized Model): \t {:.3f} seconds".format(t_3), flush=True)
        print("Step 4 (Hyper Synthesis): \t {:.3f} seconds".format(t_4), flush=True)
        print("Step 5 (Gaussian Model): \t\t {:.3f} seconds".format(t_5), flush=True)
        print("Step 6 (Hyper Synthesis): \t {:.3f} seconds".format(t_6), flush=True)
        print("Step 7 (Postprocessing): \t\t {:.3f} seconds".format(t_7), flush=True)
        print("-------------------------------------------------", flush=True)
        print("Decoding Total: \t\t {:.3f} seconds".format(t_sum), flush=True)

        t_end = time.time()
        print("Decompression time: {:.3f} sec".format(t_end - t_start))
        return final_data

    def read_bitstream(self, compressed_data):
        """ Step 1: Read the bitstream. """
        t0 = time.time()
        
        stream = BitStream()
        stream.write(compressed_data, bytes)

        y_strings, z_strings = [], []
        y_shapes, z_shapes = [], []
        points_streams, ks = [], []

        # Header
        num_frames = stream.read(np.int32)
        q_g = stream.read(np.int32)
        q_a = stream.read(np.int32)
        q = torch.tensor([[q_g, q_a]], dtype=torch.float, device=self.device)  # Example

        # Frames
        for i in range(num_frames):
            # Frame Header
            y_shape = stream.read(np.int32)
            z_shape = stream.read(np.int32)
            point_stream_length = stream.read(np.int32)
            y_stream_length = stream.read(np.int32)
            z_stream_length = stream.read(np.int32)

            k = []
            for i in range(3):
                k_level = stream.read(np.int32)
                k.append(k_level)
            
            # Content
            points = stream.read(int(point_stream_length) * 8)
            y_stream = stream.read(int(y_stream_length) * 8) # TODO: Check if we need to parse back to bytestring
            z_stream = stream.read(int(z_stream_length) * 8)
            
            # Parse Content to bytes
            points_bits = points.__str__()
            points = int(points_bits, 2).to_bytes(len(points_bits) // 8, byteorder='big')
            y_bits = y_stream.__str__()
            y_stream = int(y_bits, 2).to_bytes(len(y_bits) // 8, byteorder='big')
            z_bits = z_stream.__str__()
            z_stream = int(z_bits, 2).to_bytes(len(z_bits) // 8, byteorder='big')

            # Sort into datastructures
            ks.append(k)
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

    def hyper_synthesis_step(self, z_hats):
        """ Step 4: Hyper Synthesis """
        t0 = time.time()

        z_hat = utils.batch_sparse_tensors(z_hats, tensor_stride=32)

        gaussian_params = self.decompression_model.entropy_model.h_s(z_hat)

        t1 = time.time()
        t_step = t1 - t0
        return gaussian_params, t1 - t0

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
        # Dummy implementation
        reconstructed_pointcloud = self.decompression_model.g_s(y_hat, k=ks)
        t1 = time.time()
        print(reconstructed_pointcloud.C.shape)
        return reconstructed_pointcloud, t1 - t0

    def pack_batches(self, reconstructed_pointcloud):
        """ Step 7: Postprocessing and packing. """
        t0 = time.time()
        # Dummy implementation
        final_data = None
        t1 = time.time()
        return final_data, t1 - t0
