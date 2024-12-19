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
        z_hat, t_3 = self.factorized_model_step(z_strings, z_shapes)

        # Step 4 (Hyper Synthesis)
        gaussian_params, t_4 = self.hyper_synthesis_step(z_hat)

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
            print(i)
            # Frame Header
            y_shape = [stream.read(np.int32)]
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

            # Sort into datastructures
            ks.append(k)
            y_shapes.append(y_shape)
            y_strings.append(y_stream)
            z_strings.append(z_stream)
            points_streams.append(points)

        t1 = time.time()
        print(ks)
        print(q)
        print(y_shapes)
        return y_strings, z_strings, y_shapes, z_shapes, points_streams, ks, q, t1 - t0

    def geometry_decompression_step(self, points_streams):
        """ Step 2: Geometry Decompression """
        t0 = time.time()
        # Dummy implementation
        y_points = []
        t1 = time.time()
        return y_points, t1 - t0

    def factorized_model_step(self, z_strings, z_shapes):
        """ Step 3: Factorized Entropy Model """
        t0 = time.time()
        # Dummy implementation
        z_hat = None
        t1 = time.time()
        return z_hat, t1 - t0

    def hyper_synthesis_step(self, z_hat):
        """ Step 4: Hyper Synthesis """
        t0 = time.time()
        # Dummy implementation
        gaussian_params = None
        t1 = time.time()
        return gaussian_params, t1 - t0

    def gaussian_model_step(self, y_strings, y_shapes, y_points, q, gaussian_params):
        """ Step 5: Gaussian Entropy Model """
        t0 = time.time()
        # Dummy implementation
        y_hat = None
        t1 = time.time()
        return y_hat, t1 - t0

    def hyper_synthesis(self, y_hat, ks):
        """ Step 6: Hyper Synthesis """
        t0 = time.time()
        # Dummy implementation
        reconstructed_pointcloud = None
        t1 = time.time()
        return reconstructed_pointcloud, t1 - t0

    def pack_batches(self, reconstructed_pointcloud):
        """ Step 7: Postprocessing and packing. """
        t0 = time.time()
        # Dummy implementation
        final_data = None
        t1 = time.time()
        return final_data, t1 - t0
