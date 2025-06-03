import os
import shutil

def reorganize_images(folder1_path: str, folder2_path: str, max_frames: int = None):
    """
    Reorganizes image files from two input folders into two new output folders.
    All output images are placed directly in the base output directories (no view subdirectories).
    The output naming convention is: <frame_number>depth-<view_number>.png
    and <frame_number>rgb-<view_number>.png.

    Args:
        folder1_path (str): The path to the first input folder containing
                            'viewX/#_depth.png' files (e.g., 'view0/0_depth.png').
        folder2_path (str): The path to the second input folder containing
                            'viewX/#_masked.png' files (e.g., 'view0/0_masked.png').
        max_frames (int, optional): The maximum number of frames (starting from 0)
                                    to process per view. If None, all frames found
                                    will be processed (up to the highest frame number found).
    """

    # Define output base directories
    output_depth_base = "output/test/anon_depth"
    output_rgb_base = "output/test/anon_rgb"

    print(f"Starting image reorganization from '{folder1_path}' and '{folder2_path}'...")

    # --- Create output base directories if they don't exist ---
    # These are the only directories that will be created for output.
    try:
        os.makedirs(output_depth_base, exist_ok=True)
        os.makedirs(output_rgb_base, exist_ok=True)
        print(f"Created output directories: '{output_depth_base}' and '{output_rgb_base}'")
    except OSError as e:
        print(f"Error creating base output directories: {e}")
        return

    # --- Validate input folders ---
    if not os.path.isdir(folder1_path):
        print(f"Error: Input folder1 '{folder1_path}' does not exist or is not a directory.")
        return
    if not os.path.isdir(folder2_path):
        print(f"Error: Input folder2 '{folder2_path}' does not exist or is not a directory.")
        return

    # --- Specify the exact views you want to process ---
    # These names refer to the logical view numbers (e.g., "view0", "view1")
    views_to_process = ["view0", "view1"]

    if not views_to_process:
        print("No specific views specified for processing. Exiting.")
        return

    for view_dir_base in views_to_process: # This is the name for input and output (e.g., "view0")
        view_number_str = view_dir_base.replace("view", "")
        if not view_number_str.isdigit():
            print(f"Warning: Skipping non-standard view directory '{view_dir_base}' in the specified list. Expected 'viewX'.")
            continue
        view_number = int(view_number_str)

        print(f"\nProcessing view {view_number}...")

        # --- Construct paths for current view in input folders ---
        current_folder1_view_path = os.path.join(folder1_path, view_dir_base)
        current_folder2_view_path = os.path.join(folder2_path, view_dir_base) # This is "view0", "view1"

        # Check if the view directory actually exists in folder1
        if not os.path.isdir(current_folder1_view_path):
            print(f"Warning: View directory '{view_dir_base}' not found in '{folder1_path}'. Skipping this view.")
            continue

        # Check if the corresponding view directory exists in folder2
        if not os.path.isdir(current_folder2_view_path):
            print(f"Warning: Corresponding view directory '{view_dir_base}' not found in '{folder2_path}'. Skipping files from this view.")
            continue

        # Determine the range of frames to process
        actual_max_frames_to_process = max_frames
        if max_frames is None:
            # Find the maximum frame number in folder1's depth images for this view
            max_found_frame = -1
            for fname in os.listdir(current_folder1_view_path):
                # Now looking for #_depth.png
                if fname.endswith("_depth.png"):
                    try:
                        f_num_str = fname.split('_')[0] # Get the number before '_depth.png'
                        if f_num_str.isdigit():
                            max_found_frame = max(max_found_frame, int(f_num_str))
                    except (IndexError, ValueError):
                        continue # Malformed filename, ignore
            actual_max_frames_to_process = max_found_frame + 1 if max_found_frame >= 0 else 0
            if actual_max_frames_to_process == 0:
                print(f"No depth images found in '{current_folder1_view_path}'. Skipping frames for this view.")
                continue


        # --- Process frames from 0 up to (actual_max_frames_to_process - 1) ---
        for frame_idx in range(actual_max_frames_to_process):
            # Construct expected filenames for the current frame index in input folders
            # folder1: #_depth.png
            depth_filename_input = f"{frame_idx}_depth.png"
            # folder2: #_masked.png
            masked_filename_input = f"{frame_idx}_masked.png"

            # --- Handle Depth Image (from folder1 to substitution_depth) ---
            source_depth_path = os.path.join(current_folder1_view_path, depth_filename_input)
            # Output naming: <frame_number>depth-<view_number>.png
            dest_depth_filename = f"{frame_idx}depth-{view_number}.png"
            dest_depth_path = os.path.join(output_depth_base, dest_depth_filename)

            # --- Handle RGB (masked) Image (from folder2 to substitution_rgb) ---
            source_rgb_path = os.path.join(current_folder2_view_path, masked_filename_input)
            # Output naming: <frame_number>rgb-<view_number>.png
            dest_rgb_filename = f"{frame_idx}rgb-{view_number}.png"
            dest_rgb_path = os.path.join(output_rgb_base, dest_rgb_filename)

            # Check if both source files exist before attempting to copy
            depth_exists = os.path.exists(source_depth_path)
            rgb_exists = os.path.exists(source_rgb_path)

            if depth_exists and rgb_exists:
                try:
                    shutil.copy2(source_depth_path, dest_depth_path)
                    print(f"Copied depth: '{depth_filename_input}' to '{dest_depth_filename}'")
                    shutil.copy2(source_rgb_path, dest_rgb_path)
                    print(f"Copied RGB (masked): '{masked_filename_input}' to '{dest_rgb_filename}'")
                except Exception as e:
                    print(f"Error copying files for frame {frame_idx} in view {view_number}: {e}")
            else:
                if not depth_exists:
                    print(f"Warning: Depth file not found: '{source_depth_path}'. Skipping frame {frame_idx} for view {view_number}.")
                if not rgb_exists:
                    print(f"Warning: Masked file not found: '{source_rgb_path}'. Skipping frame {frame_idx} for view {view_number}.")

    print("\nImage reorganization complete!")

# --- Example Usage ---
if __name__ == "__main__":

    depth_images_folder = "output/xr_lubna/depth"
    masked_images_folder = "output/xr_lubna/rgb"

    reorganize_images(depth_images_folder, masked_images_folder, 10)