import LossFunction as lf
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import pandas as pd
import DataFunctions


def train(crnn, train_loader, num_epochs, device, clip_norm, optimizer, lr_scheduler, criterion, checkpoint_path,
          char2idx):
    epoch_losses = []
    iteration_losses = []
    num_updates_epochs = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        epoch_loss_list = []
        num_updates_epoch = 0
        for image_batch, text_batch in tqdm(train_loader, leave=False):
            optimizer.zero_grad()
            text_batch_logits = crnn(image_batch.to(device))
            loss = lf.compute_loss(text_batch, text_batch_logits, criterion, device, char2idx)
            iteration_loss = loss.item()

            if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                continue

            num_updates_epoch += 1
            iteration_losses.append(iteration_loss)
            epoch_loss_list.append(iteration_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(crnn.parameters(), clip_norm)
            optimizer.step()

        epoch_loss = np.mean(epoch_loss_list)
        print("Epoch:{}    Loss:{}    NumUpdates:{}".format(epoch, epoch_loss, num_updates_epoch))
        epoch_losses.append(epoch_loss)
        num_updates_epochs.append(num_updates_epoch)
        lr_scheduler.step(epoch_loss)

    torch.save(crnn.state_dict(), checkpoint_path)

    return epoch_losses, iteration_losses


def get_prediction(crnn, dataset_loader, device, idx2char):
    results = pd.DataFrame(columns=['actual', 'prediction'])
    with torch.no_grad():
        for image_batch, text_batch in tqdm(dataset_loader, leave=True):
            text_batch_logits = crnn(image_batch.to(device))  # [T, batch_size, num_classes==num_features]
            text_batch_pred = DataFunctions.decode_predictions(text_batch_logits.cpu(), idx2char)

            df = pd.DataFrame(columns=['actual', 'prediction'])
            df['actual'] = text_batch
            df['prediction'] = text_batch_pred
            results = pd.concat([results, df])
    return results.reset_index(drop=True)
