import os
import torch
import subprocess
import numpy as np
import open3d as o3d
import MinkowskiEngine as ME

gpcc_directory = "/app/dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3"

def stack_tensors(points, colors):
    """
    Stacks points and colors into PyTorch tensors.

    Points will be stacked with an additional batch/index column as the first column,
    while colors will be stacked without modification.

    Args:
        points (list of numpy.ndarray): List of N_i x 3 arrays representing 3D points.
        colors (list of numpy.ndarray): List of N_i x 3 arrays representing RGB colors.

    Returns:
        stacked_points (torch.Tensor): Tensor of shape (sum(N_i), 4) with a batch/index column.
        stacked_colors (torch.Tensor): Tensor of shape (sum(N_i), 3) without a batch/index column.
    """
    # Convert points and colors to PyTorch tensors
    points = [torch.from_numpy(p) for p in points]
    colors = [torch.from_numpy(c) for c in colors]

    # Stack points with an additional batch/index column
    stacked_points = torch.cat([
        torch.cat((torch.full((p.shape[0], 1), i, dtype=p.dtype), p), dim=1)
        for i, p in enumerate(points)
    ], dim=0)

    # Stack colors without additional columns
    stacked_colors = torch.cat(colors, dim=0)

    return stacked_points, stacked_colors


def get_points_per_batch(sparse_tensor, batch_dim=True):
    """
    Extract features grouped by batches from a MinkowskiEngine SparseTensor.

    Args:
        sparse_tensor (ME.SparseTensor): The input sparse tensor.

    Returns:
        points: A list of the points per batch.
    """
    coordinates = sparse_tensor.C  # Shape: (num_points, ndim)
    batch_indices = coordinates[:, 0]

    points_per_batch = []
    for batch_idx in batch_indices.unique():
        mask = batch_indices == batch_idx
        if batch_dim:
            points_per_batch.append(coordinates[mask, 1:])
        else:
            points_per_batch.append(coordinates[mask])

    return points_per_batch

def get_features_per_batch(features, coordinates=None):
    """
    Extract features grouped by batches from a MinkowskiEngine SparseTensor.

    Args:
        sparse_tensor (ME.SparseTensor): The input sparse tensor.

    Returns:
        features: A list of the features per batch.
    """
    # Handle optional extra coordinates
    if coordinates==None:
        coordinates = features.C  # Shape: (num_points, ndim)
    elif isinstance(features, ME.SparseTensor):
        raise ValueError("Not defined for torch features and no coordinates.")

    # Handle features being ME.SparseTensor
    if isinstance(features, ME.SparseTensor):
        features = features.F     # Shape: (num_points, feature_dim)

    batch_indices = coordinates[:, 0]

    features_per_batch = []
    for batch_idx in batch_indices.unique():
        mask = batch_indices == batch_idx
        features_per_batch.append(features[mask])

    return features_per_batch

def sort_tensor(sparse_tensor):
    """
    Sort the coordinates of a tensor

    Parameters
    ----------
    sparse_tensor: ME.SparseTensor
        Tensor containing the orignal coordinates

    returns
    -------
    sparse_tensor: ME.SparseTensor
        Tensor containing the sorted coordinates
    """
    # Sort the coordinates
    weights = torch.tensor([1e15, 1e10, 1e5, 1], dtype=torch.int64, device=sparse_tensor.device) 
    sortable_vals = (sparse_tensor.C * weights).sum(dim=1)
    sorted_coords_indices = sortable_vals.argsort()

    sparse_tensor = ME.SparseTensor(
        features=sparse_tensor.F[sorted_coords_indices],
        coordinates=sparse_tensor.C[sorted_coords_indices],
        tensor_stride=sparse_tensor.tensor_stride,
        device=sparse_tensor.device
    )
    return sparse_tensor



def sort_points(points):
    """
    Sort the coordinates of torch list sized Nx4

    Parameters
    ----------
    points: torch.tensor
        Tensor containing the orignal coordinates Nx4

    returns
    -------
    points: torch.tensor
        Tensor containing the sorted coordinates Nx4
    """
    # Sort the coordinates
    weights = torch.tensor([1e15, 1e10, 1e5, 1], dtype=torch.int64, device=points.device) 
    sortable_vals = (points * weights).sum(dim=1)
    sorted_coords_indices = sortable_vals.argsort()

    points = points[sorted_coords_indices]
    return points



def gpcc_encode(points, tmp_dir, bin_dir):
    # TODO: Make thread save (File writing operations)
    # Save as ply
    dtype = o3d.core.float32
    p_tensor = o3d.core.Tensor(points.detach().cpu().numpy()[:, 1:], dtype=dtype)
    pc = o3d.t.geometry.PointCloud(p_tensor)
    o3d.t.io.write_point_cloud(tmp_dir, pc, write_ascii=True)

    # Encode with G-PCC
    subp=subprocess.Popen(gpcc_directory + 
                    ' --mode=0' + 
                    ' --positionQuantizationScale=1' + 
                    ' --trisoupNodeSizeLog2=0' + 
                    ' --neighbourAvailBoundaryLog2=8' + 
                    ' --intra_pred_max_node_size_log2=6' + 
                    ' --inferredDirectCodingMode=0' + 
                    ' --maxNumQtBtBeforeOt=4' +
                    ' --uncompressedDataPath='+ tmp_dir + 
                    ' --compressedStreamPath='+ bin_dir, 
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
            # Read stdout and stderr
    stdout, stderr = subp.communicate()

    # Print the outputs
    if subp.returncode != 0:
        print("Error occurred:")
        print(stderr.decode())
        c=subp.stdout.readline()

    # Read the bytes to return
    with open(bin_dir, "rb") as binary:
        data = binary.read()
        
    # Clean up
    os.remove(tmp_dir)
    os.remove(bin_dir)