import pandas as pd
import torch
import torch.nn as nn
from torch.nn import LSTM
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def main():
    df = pd.read_csv(r"bias_bio.csv", index_col=[0])
    df = df.drop(columns="hard_text")
    features = df.drop(columns="gender")
    labels   = df["gender"]

    X = torch.tensor(features.values, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(labels.values,    dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 128
    num_layers  = 2
    bidir       = True
    dirs        = 2 if bidir else 1

    lstm = LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidir, dropout=0.3 if num_layers > 1 else 0.0).to(device)

    classifier = nn.Linear(hidden_size * dirs, 2).to(device)

    optimizer = torch.optim.Adam(list(lstm.parameters()) + list(classifier.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(1, epochs + 1):
        lstm.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            batch  = xb.size(0)

            h0 = torch.zeros(num_layers * dirs, batch, hidden_size, device=device)
            c0 = torch.zeros_like(h0)

            optimizer.zero_grad()
            out, _ = lstm(xb, (h0, c0))
            last    = out[:, -1, :]
            logits  = classifier(last)
            loss    = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} — avg loss: {avg_loss:.4f}")

    checkpoint = {"lstm_state_dict": lstm.state_dict(), "classifier_state_dict": classifier.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": epochs, "loss": avg_loss}
    torch.save(checkpoint, "lstm_gender_classifier.pth")
    print("Saved checkpoint to lstm_gender_classifier.pth")

    lstm.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            batch = xb.size(0)
            h0 = torch.zeros(num_layers * dirs, batch, hidden_size, device=device)
            c0 = torch.zeros_like(h0)

            out, _ = lstm(xb, (h0, c0))
            last = out[:, -1, :]
            preds = classifier(last).argmax(dim=1)
            correct += (preds == yb).sum().item()

    print(f"Training-set accuracy: {correct/len(dataset):.4f}")

if __name__ == "__main__":
    main()