import os
import torch
import open3d as o3d
from encoder.codec_single import CompressionPipeline
from decoder.codec_single import DecompressionPipeline
import numpy as np

def load_point_cloud(file_path):
    """
    Loads a .ply file using Open3D and returns points and colors as a batch.
    """
    # Load the point cloud data using Open3D
    pcd = o3d.io.read_point_cloud(file_path)

    # Extract points (Nx3) and colors (Nx3)
    points = np.asarray(pcd.points)  # (N, 3)
    colors = np.asarray(pcd.colors)  # (N, 3)

    # Normalize colors to range [0, 1] (if not already normalized)
    colors = colors / 255

    # Create a dictionary to simulate a batch item
    batch_item = {
        "points": torch.tensor(points, dtype=torch.float32),
        "colors": torch.tensor(colors, dtype=torch.float32)
    }

    # Return as a list to simulate a batch (one item batch)
    return [batch_item]


def main():
    # Input/output file paths
    input_ply = "./test.ply"
    compressed_output = "./compressed_data.bin"
    decompressed_output = "./decompressed_output.ply"

    # Check that the input PLY exists
    if not os.path.exists(input_ply):
        raise FileNotFoundError(f"Input file {input_ply} does not exist.")

    # Create encoder and decoder objects
    encoder = CompressionPipeline()
    decoder = DecompressionPipeline()

    # Load and encode the PLY file
    print("Loading PLY file...")
    batch = load_point_cloud(input_ply)
    print("PLY file loaded.")

    print("Encoding PLY file...")
    compressed_data, _ = encoder.compress(batch)
    print("Encoding completed.")

    # Now decode the compressed data
    print("Decoding compressed data...")
    final_data = decoder.decompress(compressed_data)
    print("Decoding completed.")

    # Save the decoded data to a .ply file
    print("Saving decoded PLY file...")
    o3d.io.write_point_cloud(decompressed_output, final_data)
    print(f"Decoded PLY saved to {decompressed_output}.")

    # Print the outputs (original and decoded PLY)
    print("\n=== Original PLY ===")
    with open(input_ply, "r") as f:
        print(f.read())

    print("\n=== Decoded PLY ===")
    with open(decompressed_output, "r") as f:
        print(f.read())

    # Clean up
    os.remove(compressed_output)
    os.remove(decompressed_output)

if __name__ == "__main__":
    main()
