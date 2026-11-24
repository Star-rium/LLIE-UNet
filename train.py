import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
import os
import cv2

from src.metrics import calculate_metrics, calculate_fidelity
from src.models.direct_cnn import DirectCNN
from src.models.unet import UNet
from seed import set_seed

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def apply_clahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

    return torch.from_numpy(out.astype(np.float32) / 255.0).permute(2, 0, 1)

def train(seed, train_loader, val_loader, epochs, eta, train_len, val_len,
          load=False, model_cls=DirectCNN, tag="d_cnn"):
    if load:
        ckpt = torch.load(
            f"./output/checkpoints/{tag}_seed_{seed}.pth",
            map_location=device,
            weights_only=True,
        )
        # new checkpoints are dicts; very old ones were a raw state_dict
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        model = model_cls().to(device)
        model.load_state_dict(state)
        model.eval()

        best_val = ckpt.get("best_val_loss") if "model_state_dict" in ckpt else None
        return model, None, best_val
    
    print("\n" + "=" * 60)
    print(f"START TRAINING - SEED {seed}")
    print("=" * 60)

    set_seed(seed)

    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_cls.__name__} | params: {n_params:,}")


    # experiment needed: L1, Gradient, SSIM, PSNR
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=eta
    )

    best_val_loss = float("inf")
    best_state = None

    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for low, normal in train_loader:
            low = low.to(device)
            normal = normal.to(device)

            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, normal)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * low.size(0)

        train_loss /= train_len

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low, normal in val_loader:
                low = low.to(device)
                normal = normal.to(device)
                output = model(low)

                loss = criterion(output, normal)
                val_loss += loss.item() * low.size(0)
            val_loss /= val_len

        history.append({
            "Seed": seed,
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss
        })

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Seed {seed} | "
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"Best Val: {best_val_loss:.6f}"
        )

    model.load_state_dict(best_state)
    os.makedirs("./output/checkpoints", exist_ok=True)
    torch.save(
        {
            "seed": seed,
            "model_state_dict": best_state,
            "best_val_loss": best_val_loss
        },
        f"./output/checkpoints/{tag}_seed_{seed}.pth"
    )
    return model, history, best_val_loss

METRICS = ["MLI", "CI", "E", "AG", "PSNR", "SSIM"]
METHOD_ORDER = ["Input", "Output", "Output+CLAHE", "Ground Truth"]

def evaluate_model(model, seed, test_dataset, clahe=False, clahe_clip=2.0, clahe_tiles=(8, 8)):
    print(f"\nEvaluating seed {seed}...")
    assert len(test_dataset) == 100, f"Expected 100 test images, got {len(test_dataset)}"
    model.eval()

    results = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            low, normal = test_dataset[idx]
            low_batch = low.unsqueeze(0).to(device)

            output = model(low_batch)

            output = output.squeeze(0).cpu()
            output = output.clamp(0, 1)

            methods = [
                ("Input", low),
                ("Output", output),
                ("Ground Truth", normal)
            ]
            if clahe:
                methods.insert(
                    2,
                    ("Output+CLAHE", apply_clahe(output, clahe_clip, clahe_tiles))
                )

            for method_name, image in methods:

                metrics = calculate_metrics(image)

                if method_name == "Ground Truth":
                    fidelity = {"PSNR": float("nan"), "SSIM": float("nan")}
                else:
                    fidelity = calculate_fidelity(image, normal)

                results.append({
                    "Seed": seed,
                    "Image": idx,
                    "Method": method_name,
                    **metrics,
                    **fidelity
                })

    return pd.DataFrame(results)

def summarize_metrics(results_df, ddof=1):
    """Mean ± SD across test images, per method.
    ddof=1 -> sample SD (pandas default), ddof=0 -> population SD."""
    grouped = results_df.groupby("Method")[METRICS]
    mean = grouped.mean().loc[METHOD_ORDER]
    std = grouped.std(ddof=ddof).loc[METHOD_ORDER]

    display = pd.DataFrame(index=mean.index)
    for m in METRICS:
        display[f"{m} Mean ± SD"] = [
            f"{mean.loc[k, m]:.4f} ± {std.loc[k, m]:.4f}" for k in mean.index
        ]

    # print(f"\nMETRIC SUMMARY (SD over images, ddof={ddof})")
    # print(display.to_string())
    return display