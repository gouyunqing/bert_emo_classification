from transformers import BertForSequenceClassification
import torch


model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=6,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

print(model)
