import cv2
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def fft_focus(fname, r0=20):
    if fname.lower().endswith(".fits"):
        with fits.open(fname) as hdul:
            img = hdul[0].data.astype(np.float32)
    else:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    img -= img.mean()

    fshift = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(fshift) ** 2

    height, width = img.shape
    Y, X = np.ogrid[:height, :width]
    R = np.sqrt((X - width / 2) ** 2 + (Y - height / 2) ** 2)
    mask = R > r0

    hf_energy = power[mask].sum()
    total_energy = power.sum()
    score = hf_energy / total_energy

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(img, cmap='gray', extent=[0, width, height, 0])
    axs[0].set_title('Original Image')
    axs[0].set_xlabel('Pixels (x)')
    axs[0].set_ylabel('Pixels (y)')

    u_freq = np.fft.fftshift(np.fft.fftfreq(width))
    v_freq = np.fft.fftshift(np.fft.fftfreq(height))
    extent = [u_freq[0], u_freq[-1], v_freq[0], v_freq[-1]]

    magnitude_spectrum = np.abs(fshift)
    axs[1].imshow(np.log(magnitude_spectrum), cmap='gray', extent=extent)
    axs[1].set_title('FFT Magnitude Spectrum')
    axs[1].set_xlabel('Frequency (cycles per pixel, u)')
    axs[1].set_ylabel('Frequency (cycles per pixel, v)')

    plt.tight_layout()
    plt.show()

    return score
