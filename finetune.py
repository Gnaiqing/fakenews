import torch.nn as nn
import pandas as pd
from transformers import BertConfig,BertTokenizer,BertForSequenceClassification, AdamW
from preprocess import read_fakenews,get_encoded_inputs,get_labels
from preprocess import train_path,valid_path,test_path


class BertForLiarPlus(BertForSequenceClassification):
    def __init__(self,config):
        super(BertForLiarPlus, self).__init__(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels)
        )


if __name__ == "__main__":
    # load training data
    train_df = read_fakenews(train_path)
    print("Loading data...")
    print(train_df)

    # load bert tokenizer
    print("Loading bert tokenizer...")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

    # convert data to bert format
    tokenized_inputs = get_encoded_inputs(bert_tokenizer,train_df)


    # prepare dataset

    #initialize model to finetune
    model_finetune_bert = BertForLiarPlus.from_pretrained(
        "bert-base-uncased",
        num_labels = 6,
        output_attentions = False,
        output_hidden_states = False,
    )






