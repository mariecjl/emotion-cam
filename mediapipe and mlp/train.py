import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from mlp_model import EmotionNet

#loading data
#landmarks, labels and usage purpose classification
X = np.load("X_landmarks.npy")     
y = np.load("y_labels.npy")         
usage = np.load("usage.npy")  

# conversion to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

#fer2013 usage splits
train_mask = usage == "Training"
val_mask   = usage == "PublicTest"

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

#print("Train samples:", len(X_train))
#print("Val samples:", len(X_val))

#dataloaders (validation and training)
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=64,
    shuffle=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_classes = len(torch.unique(y_train))

model = EmotionNet(num_classes=num_classes).to(device)

#class weighing for diff emotions based off of number of imgs available
class_counts = Counter(y_train.cpu().numpy())

weights = []
for i in range(num_classes):
    weights.append(1.0 / class_counts[i])

weights = torch.tensor(weights, dtype=torch.float32)
weights = weights / weights.sum() * num_classes  
weights = weights.to(device)

#print("Class weights:", weights)

#loss criterion and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

#training loop
epochs = 30

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    #validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_acc = correct / total
    scheduler.step(val_acc)

    #progress update
    print(
        f"Epoch {epoch+1:02d} "
        f"loss={avg_loss:.3f} "
        f"val_acc={val_acc:.3f}")

#saving model
torch.save(model.state_dict(), "emotion_model.pt")
print("Model saved to emotion_model.pt")

