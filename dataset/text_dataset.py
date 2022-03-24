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


class TextDatasetOriginal(Dataset):
    def __init__(self, input_file, tokenizer, **kwargs):
        # load the text data generated from before into memory
        max_input_length = kwargs.get('max_input_length', 400)
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


from itertools import groupby


def find_last(data, values):
    return [i for i, v in enumerate(data) if v in values][-1]


class TextDataset(Dataset):
    def __init__(self, input_file, tokenizer, **kwargs):
        # load the text data generated from before into memory
        max_input_length = kwargs.get('max_input_length', 400)
        text = read_input(input_file)
        print("Tokenizing and building input...")
        # tokenize the whole file
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        # generate training examples by cutting the text into blocks of size max_input_length
        self.examples = []
        self.pad = 0

        special_tokens = (tokenizer.encode('<speaker1>')[0], tokenizer.encode('<speaker2>')[0])

        block_size = max_input_length
        if block_size < 0:
            # use maximum possible input block size
            block_size = tokenizer.max_len_single_sentence
        start_pos = 0
        #for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)

        while start_pos < len(tokenized_text) - 1:
            block_of_tokens = tokenizer.build_inputs_with_special_tokens(tokenized_text[start_pos: start_pos + block_size])
            last_position_of_spec_in_block = find_last(block_of_tokens, special_tokens)
            if last_position_of_spec_in_block == 0:
                last_position_of_spec_in_block = len(block_of_tokens)
            num_to_pad = block_size - last_position_of_spec_in_block
            block_of_tokens = block_of_tokens[:last_position_of_spec_in_block] + num_to_pad * [self.pad]
            self.examples.append(block_of_tokens)
            start_pos += last_position_of_spec_in_block

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def get_data_loader(tokenizer, input_file, **kwargs):
    """ Prepare the dataset for training and evaluation """
    dataset = TextDataset(input_file, tokenizer, **kwargs)
    print("Train dataset: {:,} samples".format(len(dataset)))
    print("Build dataloaders")
    train_batch_size = kwargs.get('train_batch_size', 8)
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    return data_loader
