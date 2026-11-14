import numpy as np
import torch

def to_grayscale(image):
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    gray = (
        0.299 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.114 * image[:, :, 2]
    )

    return gray

def calculate_metrics(image):
    gray = to_grayscale(image)

    # 1. Mean Lightness / Intensity
    mli = np.mean(gray)

    # 2. Contrast Index (standard deviation)
    ci = np.std(gray)

    # 3. Shannon Entropy
    hist, _ = np.histogram(
        gray,
        bins=256,
        range=(0, 1)
    )

    probabilities = hist / hist.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    # 4. Average Gradient
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)

    dx = dx[:-1, :]
    dy = dy[:, :-1]

    gradient = np.sqrt(
        (dx ** 2 + dy ** 2) / 2
    )

    ag = np.mean(gradient)

    return {
        "MLI": float(mli),
        "CI": float(ci),
        "E": float(entropy),
        "AG": float(ag)
    }

def calculate_colorfulness(image):
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    rg = R - G
    yb = 0.5 * (R + G) - B

    std_rg = np.std(rg)
    std_yb = np.std(yb)

    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    std_root = np.sqrt(std_rg**2 + std_yb**2)
    mean_root = np.sqrt(mean_rg**2 + mean_yb**2)

    return float(std_root + 0.3 * mean_root)