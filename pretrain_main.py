from iraq_dataset import IraqEEGDataset
from model_hmms_encoder import HMMSFeatureEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def main():
    dataset = IraqEEGDataset(
        csv_path="EEG_AD_Iraq/HMMS.csv",
        binary=True
    )

    idx = list(range(len(dataset)))
    tr, te = train_test_split(idx, test_size=0.2, stratify=dataset.y)

    train_ds = torch.utils.data.Subset(dataset, tr)
    val_ds   = torch.utils.data.Subset(dataset, te)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512)

    model = HMMSFeatureEncoder()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = (torch.sigmoid(model(X)) > 0.5)
                correct += (preds == y).sum().item()
                total += y.numel()

        print(f"Epoch {epoch+1}: Val Acc = {correct/total:.4f}")

    torch.save(model.encoder.state_dict(), "hmms_encoder.pt")

if __name__ == "__main__":
    main()
