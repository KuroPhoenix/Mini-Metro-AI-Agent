#!/usr/bin/env python3
"""
bulk_resize.py

Resize all images in an input directory to 40×40 pixels and save them
to an output directory, preserving filenames.
"""

import os
from PIL import Image

# --- Configuration ---
INPUT_DIR = "stations"    # change to your source folder
OUTPUT_DIR = "templates_resized/stations_resized"  # change to your target folder
TARGET_SIZE = (40, 40)        # width, height in pixels

def ensure_output_dir(path):
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created output directory: {path}")

def is_image_file(filename):
    """Rudimentary check for common image file extensions."""
    EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return filename.lower().endswith(EXTENSIONS)

def resize_image(in_path, out_path, size):
    """Open an image, resize it (with antialiasing), and save to out_path."""
    with Image.open(in_path) as img:
        img = img.convert("RGBA") if img.mode in ("P", "LA") else img.convert("RGB")
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(out_path)
        print(f"Saved resized image: {out_path}")

def batch_resize(input_dir, output_dir, size):
    ensure_output_dir(output_dir)
    for fname in os.listdir(input_dir):
        if not is_image_file(fname):
            continue
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)
        try:
            resize_image(src, dst, size)
        except Exception as e:
            print(f"Failed to process {src}: {e}")

if __name__ == "__main__":
    batch_resize(INPUT_DIR, OUTPUT_DIR, TARGET_SIZE)
