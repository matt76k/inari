import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from inari.data import MyDataset, prepare_data_for_subgraph_task
from inari.loss import MMLoss
from inari.model import SimpleSubGMN
from inari.utils import fix_random_seed, metric_acc

fix_random_seed(42)

device = torch.device("cpu")

num_features = 35
dataset = MyDataset("data/cox2.pt", num_features)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=prepare_data_for_subgraph_task)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=prepare_data_for_subgraph_task)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=prepare_data_for_subgraph_task)

# model = SubGMN(num_features, 128, 8).to(device)
model = SimpleSubGMN(num_features, 128).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

criterion = MMLoss(mode="last")

best_score = 0

pbar = tqdm.tqdm(range(100))

for epoch in pbar:
    pbar.set_description(f"Epoch {epoch}")

    model.train()
    count = 0
    train_loss = 0

    for target, query, mm, mask in tqdm.tqdm(train_loader, leave=False, desc="Train"):
        target = target.to(device)
        query = query.to(device)
        mm = mm.to(device)
        mask = mask.to(device)

        output = model(target, query, mask)

        loss = criterion(output, mm, mask)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.detach().item()

    model.eval()
    val_loss = 0
    for target, query, mm, mask in tqdm.tqdm(val_loader, leave=False, desc="Val"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)

            output = model(target, query, mask)
            loss = criterion(output, mm, mask)

            val_loss += loss.detach().item()

    scheduler.step(val_loss)

    test_acc = 0
    for target, query, mm, mask in tqdm.tqdm(test_loader, leave=False, desc="Test"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)

            output = model(target, query, mask)
            test_acc += metric_acc(output[-1].cpu(), mm)
    test_acc /= len(test_loader)

    pbar.set_postfix(lr=optimizer.param_groups[0]["lr"], val_loss=val_loss, test_acc=test_acc)
