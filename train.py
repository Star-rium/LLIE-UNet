import torch
import torch.nn as nn 
import copy
import pandas as pd

from src.metrics import calculate_metrics
from src.models.direct_cnn import DirectCNN
from seed import set_seed

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def train(seed, train_loader, val_loader, epochs, eta, train_len, val_len):
    print("\n" + "=" * 60)
    print(f"START TRAINING - SEED {seed}")
    print("=" * 60)

    set_seed(seed)
    loader_generator = torch.Generator().manual_seed(seed)

    model = DirectCNN().to(device)


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

    torch.save(best_state, './output/d_cnn_weights.pth')
    return model, history, best_val_loss


def evaluate_model(model, seed, test_dataset):
    print(f"\nEvaluating seed {seed}...")
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

            for method_name, image in methods:

                metrics = calculate_metrics(image)

                results.append({
                    "Seed": seed,
                    "Image": idx,
                    "Method": method_name,
                    **metrics
                })

    return pd.DataFrame(results)