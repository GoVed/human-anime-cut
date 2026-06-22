import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def has_black_strip(image, threshold=9):
    """Check if left or right threshold-wide strip contains only black pixels."""
    if np.all(image[:, :threshold] == 0) or np.all(image[:, -threshold:] == 0):
        return True
    return False

def process_scenery_image(filename, input_dir, output_dir):
    """Processes a single scenery image, detecting black borders and cropping."""
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Crop image by threshold if it contains black strips
            if has_black_strip(image, threshold=48) and not has_black_strip(image, threshold=64):
                cropped = image[32:, 64:-64]
            else:
                cropped = image
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, cropped)

def filter_scenery_images(input_dir, output_dir, max_workers=4):
    """Iterates scenery preprocessing over directory in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    print(f"Preprocessing images in {input_dir} (detected {len(filenames)} files)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(
            executor.map(lambda f: process_scenery_image(f, input_dir, output_dir), filenames),
            total=len(filenames),
            desc="Preprocessing images"
        ))
    print(f"Finished preprocessing. Output images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess scenery images for style transfer GAN training")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Max workers for multi-threaded processing")
    args = parser.parse_args()

    filter_scenery_images(args.input, args.output, max_workers=args.workers)

if __name__ == "__main__":
    main()
