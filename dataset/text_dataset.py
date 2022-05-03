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


def read_lines(file):
    with open(file, 'r') as f:
        data = [line.strip() for line in f.readlines()]
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


class MessageDataset(Dataset):
    def __init__(self, input_file, tokenizer, **kwargs):
        # load the text data generated from before into memory
        max_input_length = kwargs.get('max_input_length', 400)
        text = read_lines(input_file)
        # generate training examples by cutting the text into blocks of size max_input_length
        self.examples = [tokenizer.encode(message) for message in text]
        self.examples = [ex[:max_input_length - 1] + [tokenizer.eos_token_id] for ex in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def pad_max_len(data, pad_token_id=0):
    lengths = [len(d) for d in data]
    max_len = max(lengths)
    data = torch.stack([torch.cat([d, d.new_zeros(max_len - d.size(0))], pad_token_id) for d in data])
    return data


def get_data_loader(tokenizer, input_file, mode='message', **kwargs):
    """ Prepare the dataset for training and evaluation """
    if mode == 'message':
        dataset, collate_fn = MessageDataset(input_file, tokenizer, **kwargs), pad_max_len
    else:
        dataset, collate_fn = TextDataset(input_file, tokenizer, **kwargs), None
    print(f"Dataset has: {len(dataset)} samples")
    print("Building dataloader")
    batch_size = kwargs.get('batch_size', 8)
    shuffle = kwargs.get('shuffle', True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader
