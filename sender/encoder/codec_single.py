import os
import yaml
import time
import torch
import numpy as np
import MinkowskiEngine as ME
import threading
from concurrent.futures import ThreadPoolExecutor

import utils
from unified.model import model

class CompressionPipeline:
    def __init__(self):
        # Define GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        base_path = "./unified/results/"
        self.compression_model = self.load_model(base_path)
        
    def load_model(self, base_path):
        model_name = "model_inverse_nn"
        config_path = os.path.join(base_path, model_name, "config.yaml")
        weights_path = os.path.join(base_path, model_name, "weights.pt")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        compression_model = model.ColorModel(config["model"])
        compression_model.load_state_dict(torch.load(weights_path))

        compression_model.to(self.device)
        compression_model.update()
        compression_model.eval()

        return compression_model


    def compress(self, data):
        """
        Main compression pipeline method.
        Runs all steps from analysis to bitstream writing.
        Supports multiple calls without conflict.
        """
        pointclouds, sideinfo = self.unpack_batch(data)

        # Step 1 (Analysis)
        y, k, y_points, t_1 = self.analysis_step(pointclouds) 

        # Step 2 (Hyper Analysis)
        z, t_2 = self.hyper_analysis_step(y)
        
        # Step 3 (Factorized Entropy Model)
        z_hat, z_strings, z_shapes, z_points, t_3 = self.factorized_model_step(z)

        # Step 4 (Hyper Synthesis)
        gaussian_params, t_4 = self.hyper_synthesis_step(z_hat)

        # Step 5 (Gaussian Entropy Model)
        q = torch.tensor([[1,1]], dtype=torch.float, device=self.device) # TODO: Make a parameter and add support for multiple q's
        y_strings, y_shapes, t_5 = self.gaussian_model_step(y, y_points, q, gaussian_params)

        # Step 6 (Geometry Compression with G-PCC)
        points_streams, t_6 = self.geometry_compression_step(y_points)

        # Step 7 (Bitstream Writing)
        t_7 = 0.0
        concatenated_bitstream, t_7 = self.make_bitstream(y_strings, z_strings, y_shapes, z_shapes, points_streams, k)

        t_sum = t_1 + t_2 + t_3 + t_4 + t_5 + t_6 + t_7
        print("Step 1 (Analysis): \t\t {:.3f} seconds".format(t_1), flush=True)
        print("Step 2 (Hyper Analysis): \t {:.3f} seconds".format(t_2), flush=True)
        print("Step 3 (Fact. Entr. Model): \t {:.3f} seconds".format(t_3), flush=True)
        print("Step 4 (Hyper Synthesis): \t {:.3f} seconds".format(t_4), flush=True)
        print("Step 5 (Gaussian Model): \t {:.3f} seconds".format(t_5), flush=True)
        print("Step 6 (G-PCC): \t\t\t {:.3f} seconds".format(t_6), flush=True)
        print("Step 7 (Bitstream Writer): \t {:.3f} seconds".format(t_7), flush=True)
        print("-----------------------------------------------", flush=True)
        print("Encoding Total: \t\t\t {:.3f} seconds".format(t_sum), flush=True)
        print("-----------------------------------------------", flush=True)

        return data # TODO: Mock for now 


    def unpack_batch(self, batch):
        """
        Unpacks the batch to receive torch point cloud and a list of side-info
        """
        sideinfo = []

        points, colors = [], []
        for item in batch:
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
        for z_feat, shape in zip(z_feats, shapes):
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
        
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())

        gaussian_params = utils.get_features_per_batch(gaussian_params_feats, y.C)
        y_feats = utils.get_features_per_batch(y)

        y_strings, shapes = [], []
        for y_feat, gaussian_param in zip(y_feats, gaussian_params):
            scales_hat, means_hat = gaussian_param.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            # Scale NN
            scale = self.compression_model.entropy_model.scale_nn(q) + self.compression_model.entropy_model.eps
            scale = scale[:].t().unsqueeze(0)

            indexes = self.compression_model.entropy_model.gaussian_conditional.build_indexes(scales_hat * scale)
            y_string = self.compression_model.entropy_model.gaussian_conditional.compress(
                y_feat.t().unsqueeze(0) * scale,
                indexes,
                means=means_hat * scale
            )

            y_strings.append(y_string)
            shapes.append(y_feat.shape[0])

        t1 = time.time()
        t_step = t1 - t0

        return y_strings, shapes,t_step


    def geometry_compression_step(self, y_points):
        t0 = time.time()

        base_directory = "/tmp/"

        timestamp = int(time.time() * 1e6)
        # TODO: Create unique subdirectory names for threads
        # thread_id = threading.get_ident()
        # tmp_dir = os.path.join(base_directory, f"tmp_{thread_id}_{timestamp}_points_enc.ply")
        # bin_dir = os.path.join(base_directory, f"tmp_{thread_id}_{timestamp}_points_bin.ply")

        tmp_dir = os.path.join(base_directory, f"tmp_{timestamp}_points_enc.ply")
        bin_dir = os.path.join(base_directory, f"tmp_{timestamp}_points_bin.ply")

        point_bitstreams = []
        for y_point in y_points:
            point_bitstream = utils.gpcc_encode(y_point, tmp_dir, bin_dir)
            point_bitstreams.append(point_bitstream)

        t1 = time.time()
        t_step = t1 - t0
        return point_bitstreams, t_step

    def make_bitstream(self, y_strings, z_strings, y_shapes, z_shapes, points_streams, k):
        t0 = time.time()

        

        t1 = time.time()
        t_step = t1 - t0
        return datastream


# Example usage
if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example input data
    pipeline = CompressionPipeline(batch_size=4, frame_size=10)

    # Multiple compressions in a row
    for _ in range(3):
        bitstream = pipeline.compress(data)
        print("Compression complete. Output:", bitstream)
