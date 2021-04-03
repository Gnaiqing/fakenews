import pandas as pd
import jsonlines
from transformers import BertTokenizer
from torch.utils.data import TensorDataset

def read_fakenews(fakenews_path):
    """
    read fakenews data from file
    :param fakenews_path: input path
    :return: pandas DataFrame containing all the information
    """
    fakenews = []
    label_dict = {
        "pants-fire":0,
        "false":1,
        "barely-true":2,
        "half-true":3,
        "mostly-true":4,
        "true":5
    }
    with jsonlines.open(fakenews_path) as reader:
        for obj in reader:
            obj["label"] = label_dict[obj["label"]]
            fakenews.append(obj)
    fakenews_df = pd.DataFrame(fakenews)
    # print(fakenews_df.head())
    return fakenews_df


def get_encoded_inputs(tokenizer, fakenews_df):
    """
    Convert Dataframe containing Fakenews to transformer inputs
    We currently use the claim and evidence only
    :param fakenews_df: Dataframe containing Fakenews data
    :return: encoded inputs
    """

    batch_claim = fakenews_df["claim"].astype(str).tolist()
    batch_justification = fakenews_df["justification"].astype(str).tolist()
    encoded_inputs = tokenizer(batch_claim,
                               batch_justification,
                               padding=True,
                               truncation='only_second',
                               return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_masks = encoded_inputs['attention_mask']
    print("Showing tokenized input...")
    print("Tokenized:",tokenizer.decode(input_ids[0]))
    print("Attention_mask:",attention_masks[0])
    return encoded_inputs


def get_labels(fakenews_df):
    """
    Get labels from Dataframe containing Fakenews
    :param fakenews_df: Dataframe containing Fakenews data
    :return: list containing labels
    """
    labels = fakenews_df["label"].astype(int).tolist()
    print("Showing labels...")
    print(labels[0:5])
    return labels


def create_dataset(tokenizer, fakenews_path):
    """
    Create dataset from jsonl file in fakenews_path
    :param tokenizer: Tokenizer used
    :param fakenews_path: path for fakenews jsonl file
    :return: TensorDataset for fakenews
    """
    fakenews_df = read_fakenews(fakenews_path)
    encoded_inputs = get_encoded_inputs(tokenizer,fakenews_df)
    labels = get_labels(fakenews_df)
    input_ids = encoded_inputs['input_ids']
    attention_masks = encoded_inputs['attention_mask']
    dataset = TensorDataset(input_ids,attention_masks,labels)
    return dataset


train_path = "LIAR-PLUS/dataset/jsonl/train2.jsonl"
valid_path = "LIAR-PLUS/dataset/jsonl/val2.jsonl"
test_path = "LIAR-PLUS/dataset/jsonl/test2.jsonl"

if __name__ == "__main__":
    df = read_fakenews(test_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    encoded_inputs = get_encoded_inputs(tokenizer, df)
    for ids in encoded_inputs["input_ids"]:
        print(tokenizer.decode(ids))