import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import torch
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from config import Config


def load_txt(txt_path):
    sent_list = []
    label_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sent_label = line.strip().split(';')
            sent = sent_label[0].lower()
            label = sent_label[1]
            sent_list.append(sent)
            label_list.append(label)

    return sent_list, label_list


def encode_sent(sent_list, tokenizer, MAX_LEN):
    input_ids = []
    for sent in sent_list:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )
        input_ids.append(encoded_sent)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')
    return input_ids


def encode_label(label_list):
    label2label_idx = {
        'sadness': 0,
        'anger': 1,
        'love': 2,
        'joy': 3,
        'fear': 4,
        'surprise': 5
    }
    label_ids = []
    for label in label_list:
        label_ids.append(int(label2label_idx[label]))
    return label_ids


def generate_mask(input_ids):
    attention_masks = []
    for sent in input_ids:
        mask = [int(token_id) > 0 for token_id in sent]
        attention_masks.append(mask)
    return attention_masks


def build_dataloader(input_ids, label_ids, masks, is_train, batch_size):
    data = TensorDataset(input_ids, masks, label_ids)
    if is_train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def load_data(train_path, val_path, test_path, MAX_LEN=64, batch_size=64):
    train_sent_list, train_label_list = load_txt(train_path)
    val_sent_list, val_label_list = load_txt(val_path)
    test_sent_list, test_label_list = load_txt(test_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_input_ids = encode_sent(train_sent_list, tokenizer, MAX_LEN)
    val_input_ids = encode_sent(val_sent_list, tokenizer, MAX_LEN)
    test_input_ids = encode_sent(test_sent_list, tokenizer, MAX_LEN)

    train_label_ids = torch.tensor(encode_label(train_label_list))
    val_label_ids = torch.tensor(encode_label(val_label_list))
    test_label_ids = torch.tensor(encode_label(test_label_list))

    train_masks = torch.tensor(generate_mask(train_input_ids))
    val_masks = torch.tensor(generate_mask(val_input_ids))
    test_masks = torch.tensor(generate_mask(test_input_ids))

    train_input_ids = torch.tensor(train_input_ids)
    val_input_ids = torch.tensor(val_input_ids)
    test_input_ids = torch.tensor(test_input_ids)

    train_dataloader = build_dataloader(train_input_ids, train_label_ids, train_masks, is_train=True, batch_size=batch_size)
    val_dataloader = build_dataloader(val_input_ids, val_label_ids, val_masks, is_train=False, batch_size=batch_size)
    test_dataloader = build_dataloader(test_input_ids, test_label_ids, test_masks, is_train=False, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = load_data(train_path='./data/train.txt',
                                                                    val_path='./data/val.txt',
                                                                    test_path='./data/test.txt')










