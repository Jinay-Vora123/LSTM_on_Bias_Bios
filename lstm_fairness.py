import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

df = pd.read_csv(r"bias_bio.csv", index_col=[0])
texts = df['hard_text'].astype(str).tolist()
df = df.drop(columns="hard_text")
sd = StandardDataset(df=df, label_name='gender', favorable_classes=[1], protected_attribute_names=['profession'], privileged_classes=[[df['profession'].value_counts().idxmax()]], categorical_features=[])

rw = Reweighing(unprivileged_groups=[{'profession': p} for p in df['profession'].unique() if p!=sd.privileged_protected_attributes[0][0]], privileged_groups=[{'profession': sd.privileged_protected_attributes[0][0]}])
rw.fit(sd)
sd_transf = rw.transform(sd)

tok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased').eval()

def embed_texts(texts, batch_size=32, max_len=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert.to(device)
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = bert(**enc).last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1)
            summed = (out * mask).sum(1)
            counts = mask.sum(1)
            emb = (summed / counts).cpu()
        all_emb.append(emb)
    return torch.cat(all_emb, dim=0)

X = embed_texts(texts)
y = sd_transf.labels.ravel()
w = sd_transf.instance_weights

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42, stratify=y)

class OneStepLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        out,(hn,_) = self.lstm(x)
        return self.fc(hn[-1])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OneStepLSTM(X.size(1), 128, 1, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_dataset = torch.utils.data.TensorDataset(X_train, torch.tensor(y_train, dtype=torch.long), torch.tensor(w_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(5):
    model.train()
    total_loss = 0
    for xb, yb, wb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss = (loss * wb).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} loss: {total_loss/len(train_loader):.4f}")

torch.save({'model_state_dict': model.state_dict(), 'tokenizer_name': 'bert-base-uncased'}, 'fair_lstm_gender_classifier.pth')