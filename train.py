import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from model import model
import random
from load_data import load_data


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    label_flat = labels.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)


def train(model, config):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.eps)
    epoch = config.epoch
    train_dataloader, val_dataloader, test_dataloader = load_data(train_path=config.train_path,
                                                                  val_path=config.val_path,
                                                                  test_path=config.test_path,
                                                                  MAX_LEN=config.max_len,
                                                                  batch_size=config.batch_size)
    total_steps = len(train_dataloader) * epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_values = []

    for epoch_i in range(0, epoch):
        print('')
        print('=========== Epoch {:}/{:} =========='.format(epoch_i + 1, epoch))
        print('train...............')

        model.train()
        total_loss = 0

        for i, batch in enumerate(train_dataloader):
            if i % 50 == 0 and i != 0:
                print('  Batch {:>5,}  of  {:>5,}'.format(i, len(train_dataloader)))
            b_input_ids = batch[0].to(device)
            b_masks = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)
            loss = output[0]

            total_loss += loss.item()

            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss/len(train_dataloader)
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        model.eval()
        val_loss, val_acc = 0, 0
        n_eval_steps = 0

        for batch in val_dataloader:
            val_input_ids = batch[0].to(device)
            val_masks = batch[1].to(device)
            val_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(val_input_ids,
                                token_type_ids=None,
                                attention_mask=val_masks)
            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            val_labels = val_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, val_labels)
            val_acc += tmp_eval_accuracy
            n_eval_steps += 1
        print("  Accuracy: {0:.2f}".format(val_acc / n_eval_steps))

    print("")
    print("Training complete!")










