
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from .validate import validate

def train(model: nn.Module, train_loader, val_loader, num_epochs, optimizer, criterion, 
        logger, tb_writer, cp_name, device="cuda", reload_best=True, show_grad=False,
        use_scheduler=False):
    
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_loss = 1e9
    best_epoch = 0
    model.to(device)
    for epoch in range(num_epochs):
        ## training step
        model.train()
        train_loss = 0
        correct = 0
        for idx, batch in enumerate(train_loader):
            inputs = batch["data"].float().to(device)
            labels = batch["label"].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()        
            _, curr_pred = torch.max(outputs, 1)
            correct += torch.eq(curr_pred, labels).sum(-1).item()
        train_loss /= (idx + 1)
        train_acc = correct/len(train_loader.dataset)

        ## validation step
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

                _, curr_pred = torch.max(outputs, 1)
                correct += torch.eq(curr_pred, labels).sum(-1).item()
        val_loss /= (idx + 1)
        val_acc = correct/len(val_loader.dataset)

        if show_grad:
            for p in model.parameters():
                print(torch.min(p.grad), torch.max(p.grad))

        if use_scheduler:
            scheduler.step(val_loss)

        msg = "Epoch: {} \tTrain Loss: {:.6f} \tVal Loss: {:.6f} \tTrain Acc: {:.6f} \tVal Acc: {:.6f}"
        msg = msg.format(epoch, train_loss, val_loss, train_acc, val_acc)
        logger.info(msg)

        tb_writer.add_scalar("train/loss", train_loss, epoch+1)
        tb_writer.add_scalar("val/loss", val_loss, epoch+1)
        tb_writer.add_scalar("val/accuracy", val_acc, epoch+1)

        if val_loss <= best_loss:
            ## save checkpoint
            state = {"epoch":epoch,
                     "model_state_dict":model.state_dict(),
                     "optimizer_state_dict":optimizer.state_dict(),
                     "train_loss": train_loss,
                     "val_loss": val_loss,
                     "val_acc": val_acc}
            torch.save(state, cp_name)
            best_loss = val_loss
            best_epoch = epoch

    if reload_best:
        msg = "Training done. Reloading checkpoint from Epoch {}."
        msg = msg.format(best_epoch)
        logger.info(msg)

        checkpoint = torch.load(cp_name, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

    ## get final loss
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    msg = "Final Val Loss: {:.6f} \tFinal Val Acc: {:.6f}"
    msg = msg.format(val_loss, val_acc)
    logger.info(msg)

    return model, best_epoch


    

    