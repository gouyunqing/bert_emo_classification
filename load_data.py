import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, Dataset, DataLoader
import torch
from transformers import BertTokenizer


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


def encode_sent(sent_list, tokenizer):
    input_ids = []
    for sent in sent_list:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
        )
        input_ids.append(encoded_sent)
    return input_ids


def encode_label(label_list):
    label2label_idx = {
        'sadness': 0,
        'anger': 1,
        'love': 2,
        
    }


def load_data(train_path, val_path, test_path):
    train_sent_list, train_label_list = load_txt(train_path)
    val_sent_list, val_label_list = load_txt(val_path)
    test_sent_list, test_label_list = load_txt(test_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_input_ids = encode_sent(train_sent_list, tokenizer)
    val_input_ids = encode_sent(val_sent_list, tokenizer)
    test_input_ids = encode_sent(test_sent_list, tokenizer)








