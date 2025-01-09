
import torch

def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0
    correct = 0
    
    with torch.no_grad(): 
        for idx, batch in enumerate(val_loader):
            inputs = batch["data"].float().to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            correct += torch.eq(predictions, labels).sum(-1).item()
    
    val_loss /= (idx + 1)
    val_acc = correct/len(val_loader.dataset)
    return val_loss, val_acc