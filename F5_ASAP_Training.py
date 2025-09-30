import torch
import torch.nn as nn
import torch.optim as optim
from F3_ASAP_Transform import train_loader, val_loader
from F4_ASAP_Model import StomataNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StomataNet().to(device)
criterion_cls = nn.BCEWithLogitsLoss()
criterion_reg = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
best_val = float('inf')
for epoch in range(1, 51):
    model.train()
    total_loss = 0.0
    for imgs, is_op, apr in train_loader:
        imgs = imgs.to(device)
        is_op = is_op.float().to(device)
        apr = apr.float().to(device)
        optimizer.zero_grad()
        logits, preds = model(imgs)
        # Classification: target = is_op (1 aperture, 0 NC)
        loss_c = criterion_cls(logits, is_op.unsqueeze(1))
        # Regression: only if is_op==1
        mask = is_op.bool()
        if mask.any():
            loss_r = criterion_reg(preds[mask], apr[mask].unsqueeze(1))
        else:
            loss_r = torch.tensor(0.0, device=device)
        loss = loss_c + loss_r
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, is_op, apr in val_loader:
            imgs = imgs.to(device)
            is_op = is_op.float().to(device)
            apr = apr.float().to(device)
            logits, preds = model(imgs)
            loss_c = criterion_cls(logits, is_op.unsqueeze(1))
            mask = is_op.bool()
            loss_r = criterion_reg(preds[mask], apr[mask].unsqueeze(1)) if mask.any() else torch.tensor(0.0, device=device)
            val_loss += (loss_c + loss_r).item()

    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f'Epoch {epoch}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}')
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), 'stomata_model.pt')

