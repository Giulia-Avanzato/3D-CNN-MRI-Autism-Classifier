import torch


def check_accuracy(net, loader, device="cpu"):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0.0
    return accuracy


def train_and_save(net, epochs, criterion, optimizer, train_loader,
                    test_loader, save_path=None, device="cpu",
                    verbose=True):
    net.to(device)
    best_accuracy = 0.0
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = check_accuracy(net, test_loader, device=device)

        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_accuracy)

        if verbose:
            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Loss: {epoch_loss:.4f} "
                  f"Accuracy: {epoch_accuracy:.2f}%")

        if save_path is not None and epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(net.state_dict(), save_path)
            if verbose:
                print(f"  -> Nuovo best model salvato in "
                      f"{save_path} (accuracy {best_accuracy:.2f}%)")

    history["best_accuracy"] = best_accuracy
    return history
