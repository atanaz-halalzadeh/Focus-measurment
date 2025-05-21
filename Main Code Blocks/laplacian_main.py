import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from glob import glob

def analyze_laplacian_from_folder(folder_path, lap_var_threshold=100.0, max_var=500.0):
    """
    Analyze all images (FITS, PNG, JPG) in a folder using Laplacian sharpness metrics.
    """
    focus_values = []

    valid_exts = ['.fits', '.fit', '.fts', '.png', '.jpg', '.jpeg', '.bmp']

    image_files = [
        f for f in glob(os.path.join(folder_path, '*'))
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    if not image_files:
        print("❌ No valid image files found in folder.")
        return []

    for image_path in image_files:
        focus = analyze_laplacian_single_image(
            image_path,
            lap_var_threshold=lap_var_threshold,
            max_var=max_var
        )
        if focus is not None:
            focus_values.append(focus)

    return focus_values


def analyze_laplacian_single_image(image_path, lap_var_threshold=100.0, max_var=500.0):
    """
    Analyzes a single image file using Laplacian sharpness metrics and plots results.
    Returns Laplacian variance (focus value).
    """
    filename = os.path.basename(image_path)
    ext = os.path.splitext(image_path)[1].lower()

    # Load image
    if ext in [".fits", ".fit", ".fts"]:
        try:
            with fits.open(image_path) as hdul:
                data = hdul[0].data
            if data is None:
                raise ValueError("Empty FITS data.")
            data = np.nan_to_num(data)
            data = data - np.min(data)
            if np.max(data) != 0:
                data = data / np.max(data)
            image = (data * 255).astype(np.uint8)
        except Exception as e:
            print(f"❌ Error loading FITS file '{image_path}': {e}")
            return None
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ Error: Could not load image at '{image_path}'")
            return None

    # Apply Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    laplacian_img = np.uint8(np.clip(laplacian_abs, 0, 255))

    # Metrics
    lap_focus = laplacian.var()
    lap_blur_score = min(lap_focus / max_var, 1.0)
    lap_class = "Sharp" if lap_focus >= lap_var_threshold else "Blur"

    # Print Results
    print(f"📂 {filename}")
    print(f"  🔍 Focus (Laplacian Var): {lap_focus:.2f}")
    print(f"  🧠 Class: {lap_class}")
    print(f"  🎯 Confidence: {lap_blur_score * 100:.1f}%\n")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.suptitle(f"{filename} | {lap_class} ({lap_blur_score * 100:.1f}%)", fontsize=13)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Laplacian (Edges)")
    plt.imshow(laplacian_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    return lap_focus

