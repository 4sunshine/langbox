"""
HEAVILY BASED ON:
https://github.com/mar-muel/artificial-self-AMLD-2020/tree/master/2
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


def read_input(file):
    with open(file, 'r') as f:
        data = f.read()
    return data


class TextDataset(Dataset):
    def __init__(self, input_file, tokenizer, max_input_length=400):
        # load the text data generated from before into memory
        text = read_input(input_file)
        print("Tokenizing and building input...")
        # tokenize the whole file
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        # generate training examples by cutting the text into blocks of size max_input_length
        self.examples = []
        block_size = max_input_length
        if block_size < 0:
            # use maximum possible input block size
            block_size = tokenizer.max_len_single_sentence
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def get_data_loader(tokenizer, input_file, **kwargs):
    """ Prepare the dataset for training and evaluation """
    dataset = TextDataset(tokenizer, input_file, **kwargs)
    print("Train dataset: {:,} samples".format(len(dataset)))
    print("Build dataloaders")
    train_batch_size = kwargs.get('train_batch_size', 8)
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    return data_loader
