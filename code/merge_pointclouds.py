import open3d as o3d
import open3d.core as o3c
import open3d.t.io as o3tio
import open3d.t.geometry as o3tgeo
import numpy as np
import os
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import torch
import math

def get_intrinsic(resolution):
    """Creates the camera intrinsic parameter based on resolution"""
    if resolution == '720p':
        intrinsic_matrix = o3c.Tensor([[530.51544, 0, 646.29645],
                                       [0, 530.51544, 341.9635],
                                       [0, 0, 1]], dtype=o3c.float64)
    elif resolution == '1080p':
        intrinsic_matrix = o3c.Tensor([[1078.7, 0, 949],
                                       [0, 1078.49, 563.243],
                                       [0, 0, 1]], dtype=o3c.float64)
    elif resolution == '2K':
        intrinsic_matrix = o3c.Tensor([[915.138375, 0, 1103.5],
                                       [0, 915.138375, 620.5],
                                       [0, 0, 1]], dtype=o3c.float64)
    elif resolution == '4K':
        intrinsic_matrix = o3c.Tensor([[1591.545, 0, 1919.5],
                                       [0, 1591.545, 1079.5],
                                       [0, 0, 1]], dtype=o3c.float64)
    else:
        raise ValueError("Unsupported resolution")
    return intrinsic_matrix

def convert_to_point_cloud(color_tensor, depth_tensor, intrinsic_tensor):
    """Converts RGBD image to a point cloud"""
    # Create RGBD image
    rgbd_image = o3tgeo.RGBDImage(color=color_tensor, depth=depth_tensor)
    
    # Create point cloud from RGBD image
    point_cloud = o3tgeo.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic_tensor,
        depth_scale=1000.0,
        depth_max=1000.0,
    )
    
    return point_cloud

def convert_color_values(pcd):
    """Converts normalized color values [0, 1] to 0-255 range"""
    colors = pcd.point["colors"]
    colors = (colors * 255).to(o3c.Dtype.UInt8)
    pcd.point["colors"] = colors
    return pcd

def load_transformation_matrices():
    """Loads transformation matrices from text files"""
    matrices = {}
    transforms_dir = 'transforms'

    for index in range(num_views):
        file_path = os.path.join(transforms_dir, f'view-{index}.txt')
        with open(file_path, 'r') as file:
            lines = file.readlines()
            matrix = np.array([[float(value) for value in line.split()] for line in lines])
            matrices[index] = matrix

    return matrices


def load_random_synthetic_human():
    # Randomly pick from pre-generated point clouds
    candidates = [
        "substitution_humans/person1.ply"
    ]
    chosen = random.choice(candidates)
    human = o3d.io.read_point_cloud(chosen)
    tensor_cloud = o3d.t.geometry.PointCloud.from_legacy(human)

    colors = tensor_cloud.point['colors']
    tensor_cloud.point['colors'] = colors.to(o3d.core.Dtype.UInt8)
    return tensor_cloud

def process_views_in_batch_substitution(view_batch, intrinsic, matrices):
    clouds = []
    read_times = []
    mask_removal_times = []

    for rgb_path, depth_path, view_index in view_batch:
        # RGBD Image Read Phase
        read_start = time.perf_counter()
        color_tensor = o3tio.read_image(rgb_path).cuda(device_id=0)
        depth_tensor = o3tio.read_image(depth_path).cuda(device_id=0)
        read_time = time.perf_counter() - read_start

        # Mask Removal Phase
        mask_removal_start = time.perf_counter()
        cloud = convert_to_point_cloud(color_tensor, depth_tensor, intrinsic).cuda(device_id=0)
        cloud = convert_color_values(cloud)

        positions = cloud.point['positions']
        colors = cloud.point['colors']

        private_mask = (colors[:, 0] == 0) & (colors[:, 1] == 255) & (colors[:, 2] == 0)

        person_points = positions[private_mask]
        mask_inv = private_mask.logical_not()

        cloud.point['positions'] = positions[mask_inv]
        cloud.point['colors'] = colors[mask_inv]

        # Insert new synthetic humans for each red-masked region
        if person_points.shape[0] > 0:
            # GPU: Get bounding box and center of each red-masked region
            private_mask = (colors[:, 0] == 0) & (colors[:, 1] == 255) & (colors[:, 2] == 0)
            private_mask_reshaped = private_mask.reshape(-1)
            unique_masks = private_mask_reshaped.nonzero(as_tuple=False)  # Get indices of all red-masked pixels

            # Iterate over each red-masked region
            for mask_idx in unique_masks:
                # Get the points corresponding to the current red-masked region
                region_points = positions[mask_idx]

                # GPU: Get bounding box and center for this specific region
                min_pos = region_points.min(0)
                max_pos = region_points.max(0)
                center = region_points.mean(0)
                diff = max_pos - min_pos
                scale_factor = (diff * diff).sum().sqrt()

                # Load + prepare synthetic human
                synthetic_human = load_random_synthetic_human().cuda(device_id=0)
                synthetic_positions = synthetic_human.point['positions']

                # GPU: Compute synthetic human size (bounding box diag)
                min_syn = synthetic_positions.min(0)
                max_syn = synthetic_positions.max(0)
                syn_diff = max_syn - min_syn
                synthetic_size = (syn_diff * syn_diff).sum().sqrt()

                # Final scale factor (small epsilon to avoid div-by-zero)
                scale = scale_factor / (synthetic_size + 1e-6)

                # Fast rotation (optional: keep fixed for all frames)
                angle = random.uniform(0, 2 * np.pi)
                rotation_matrix = o3c.Tensor([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ], dtype=o3c.Dtype.Float32, device=synthetic_positions.device)

                # GPU: Apply scale, rotation, and translation
                positions_scaled = synthetic_positions * scale
                positions_rotated = positions_scaled @ rotation_matrix.T()
                synthetic_human.point['positions'] = positions_rotated + center

                # Append synthetic human to the cloud
                cloud += synthetic_human

        cloud, _ = cloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
        cloud.transform(matrices[view_index])
        mask_removal_time = time.perf_counter() - mask_removal_start

        clouds.append(cloud)
        read_times.append(read_time)
        mask_removal_times.append(mask_removal_time)

    return clouds, read_times, mask_removal_times


def process_views_in_batch(view_batch, intrinsic, matrices):
    """Processes a batch of views by reading in color and depth PNGs, converting to point clouds, denormalizing the RGB values, and applying the transformation matrix."""
    clouds = []
    read_times = []
    mask_removal_times = []
    
    for rgb_path, depth_path, view_index in view_batch:
        # RGBD Image Read Phase
        read_start = time.perf_counter()
        color_tensor = o3tio.read_image(rgb_path).cuda(device_id=0)
        depth_tensor = o3tio.read_image(depth_path).cuda(device_id=0)
        read_time = time.perf_counter() - read_start

        # Mask Removal Phase
        mask_removal_start = time.perf_counter()
        cloud = convert_to_point_cloud(color_tensor, depth_tensor, intrinsic).cuda(device_id=0)
        cloud = convert_color_values(cloud)

        positions = cloud.point['positions']
        colors = cloud.point['colors']

        private_mask = (colors[:, 0] == 255) & (colors[:, 1] == 0) & (colors[:, 2] == 0)
        positions[private_mask, 2] = np.inf
        cloud.point['positions'] = positions

        cloud, _ = cloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
        cloud.transform(matrices[view_index])
        mask_removal_time = time.perf_counter() - mask_removal_start

        clouds.append(cloud)
        read_times.append(read_time)
        mask_removal_times.append(mask_removal_time)
    
    return clouds, read_times, mask_removal_times

def process_frame(rgb_dir, depth_dir, output_dir, frame_number, view_indices, matrices, resolution, batch_size=2):
    """Processes all views of a single frame and merges them into a single point cloud using batching."""
    intrinsic = get_intrinsic(resolution)
    merged_cloud = None
    
    read_times = []
    mask_removal_times = []
    
    # Processing Phase
    process_start = time.perf_counter()

    view_batch = []
    for view_index in view_indices:
        rgb_path = os.path.join(rgb_dir, f"{frame_number}rgb-{view_index}.png")
        depth_path = os.path.join(depth_dir, f"{frame_number}depth-{view_index}.png")
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"Missing file: {rgb_path} or {depth_path}")
            continue
        view_batch.append((rgb_path, depth_path, view_index))
    
    # Processing in batches
    batch_results = []
    for i in range(0, len(view_batch), batch_size):
        batch = view_batch[i:i + batch_size]
        if substitution:
            clouds, batch_read_times, batch_mask_removal_times = process_views_in_batch_substitution(batch, intrinsic, matrices)
        else:
            clouds, batch_read_times, batch_mask_removal_times = process_views_in_batch(batch, intrinsic, matrices)
        batch_results.append((clouds, batch_read_times, batch_mask_removal_times))
    batch_time = time.perf_counter() - process_start

    # Merge Phase
    merge_start = time.perf_counter()
    for clouds, batch_read_times, batch_mask_removal_times in batch_results:
        read_times.extend(batch_read_times)
        mask_removal_times.extend(batch_mask_removal_times)
        
        for cloud in clouds:
            if merged_cloud is None:
                merged_cloud = cloud
            else:
                merged_cloud += cloud
    merge_time = time.perf_counter() - merge_start  # Measure time for merging clouds

    flip_start = time.perf_counter()
    # Apply transformation to flip the point cloud so it is not upside down
    flip_transform = np.array([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])
    merged_cloud.transform(flip_transform)
    flip_time = time.perf_counter() - flip_start
    
    process_time = time.perf_counter() - process_start  # Total time for processing the frame
    
    # Point Cloud Writing Phase
    write_start = time.perf_counter()
    output_path = os.path.join(output_dir, f"mergedframe-{frame_number}.ply")
    o3tio.write_point_cloud(output_path, merged_cloud)
    write_time = time.perf_counter() - write_start  # Measure time for writing output
    
    avg_frame_read_time = sum(read_times)
    avg_view_mask_removal_time = sum(mask_removal_times)
    
    return avg_frame_read_time, avg_view_mask_removal_time, batch_time, merge_time, flip_time, process_time, write_time

def merge_point_clouds_parallel(rgb_dir, depth_dir, output_dir, matrices, resolution):
    """Processes multiple frames in parallel using multiprocessing"""
    print("Merging point clouds in parallel...")
    
    rgb_files = [f for f in os.listdir(rgb_dir) if 'rgb' in f and f.endswith('.png')]
    frames = set(f.split('rgb')[0] for f in rgb_files)

    read_times = []
    mask_removal_times = []
    batch_times = []
    merge_times = []
    flip_times = []
    process_times = []
    write_times = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_frame, rgb_dir, depth_dir, output_dir, frame_number, range(num_views), matrices, resolution): frame_number
                   for frame_number in sorted(frames)}
        for future in as_completed(futures):
            avg_frame_read_time, avg_view_mask_removal_time, batch_time, merge_time, flip_time, process_time, write_time = future.result()
            read_times.append(avg_frame_read_time)
            mask_removal_times.append(avg_view_mask_removal_time)
            batch_times.append(batch_time)
            merge_times.append(merge_time)
            flip_times.append(flip_time)
            process_times.append(process_time)
            write_times.append(write_time)
    
    return read_times, mask_removal_times, batch_times, merge_times, flip_times, process_times, write_times

def main():

    if anon:
        output_dir = 'output/merge_points_test/anon_frames'
        rgb_dir = 'output/merge_points_test/anon_rgb'
        depth_dir = 'output/merge_points_test/anon_depth'
    else:
        if demo:
            output_dir = 'output/merge_points_test/test_frames'
            rgb_dir = 'output/merge_points_test/test_rgb'
            depth_dir = 'output/merge_points_test/test_depth'
        else:
            output_dir = 'data_path/data_frames_jc_longdress_n2'
            rgb_dir = 'data_path/data_rgb_jc_longdress_n2'
            depth_dir = 'data_path/data_depth'
        
        if substitution and demo:
            output_dir = 'output/merge_points_test/substitution_frames'

        if noise:
            depth_dir += "_noise"
        elif not demo and not substitution:
            depth_dir += "_jc_longdress"


    resolution = '720p'

    os.makedirs(output_dir, exist_ok=True)
    
    matrices = load_transformation_matrices()

    def execute():
        read_times, mask_removal_times, batch_times, merge_times, flip_times, process_times, write_times = merge_point_clouds_parallel(
            rgb_dir, depth_dir, output_dir, matrices, resolution)
        return read_times, mask_removal_times, batch_times, merge_times, flip_times, process_times, write_times
    
    parallel_start = time.perf_counter()
    read_times, mask_removal_times, batch_times, merge_times, flip_times, process_times, write_times = execute()
    parallel_time_io = time.perf_counter() - parallel_start

    avg_read_time = sum(read_times) / len(read_times)
    avg_mask_removal_time = sum(mask_removal_times) / len(mask_removal_times)
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_merge_time = sum(merge_times) / len(merge_times)
    avg_flip_time = sum(flip_times) / len(flip_times)
    avg_process_time = sum(process_times) / len(process_times)
    avg_write_time = sum(write_times) / len(write_times)

    e2e_latency_io = avg_process_time + avg_write_time
    e2e_latency = avg_mask_removal_time + avg_merge_time + avg_flip_time

    parallel_time = parallel_time_io * e2e_latency / e2e_latency_io
    
    frame_time_io = (parallel_time_io - e2e_latency_io) / (frames - 1)
    frame_time = (parallel_time - e2e_latency) / (frames - 1)
    
    print(f"Average Frame Read Time: {avg_read_time:.3f} seconds")
    print(f"Average Mask Removal Time: {avg_mask_removal_time:.3f} seconds")
    print(f"Average Batch Time: {avg_batch_time:.3f} seconds")
    print(f"Average Merge Time: {avg_merge_time:.3f} seconds")
    print(f"Average Flip Time: {avg_flip_time:.3f} seconds")
    print(f"Average Processing Time: {avg_process_time:.3f} seconds")
    print(f"Average Write Time: {avg_write_time:.3f} seconds\n")

    print(f"End-to-end Latency w/ IO: {e2e_latency_io:.3f} seconds")
    print(f"Parallel Total Processing Time w/ IO: {parallel_time_io:.3f} seconds")
    print(f"Frame Processing Time w/ IO: {frame_time_io:.3f} seconds")
    print(f"Frames Per Second w/ IO: {1 / frame_time_io:.3f} FPS\n")

    print(f"End-to-end Latency w/o IO: {e2e_latency:.3f} seconds")
    print(f"Parallel Total Processing Time w/o IO: {parallel_time:.3f} seconds")
    print(f"Frame Processing Time w/o IO: {frame_time:.3f} seconds")
    print(f"Frames Per Second w/o IO: {1 / frame_time:.3f} FPS")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    noise = False
    frames = 50 if demo else 50
    num_views = 2

    # Running with synthetic data or real
    demo = True
    # experimental, output not currently fully correct
    substitution = False
    # if running the anonymous pipeline
    anon = True
    
    main()
