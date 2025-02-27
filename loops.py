import torch


def train_loop(dataloader, model, loss_fn, optimizer, device, BATCH_SIZE):
    size = len(dataloader.dataset)
    model.train()

    loss_tracking = []
    accuracies = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # loss of pred
        pred = model(X)
        loss = loss_fn(pred, y)
        accuracies.append((pred.argmax(1) == y).type(torch.float).sum().item() / BATCH_SIZE)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_tracking.append(loss.item())
        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")

    return loss_tracking, accuracies


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
