import os
import sys

import torch
import numpy as np
import random


ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ('<speaker1>', '<speaker2>')}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def split_train_val(text_file, validation_part=0.1):
    with open(text_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    total_lines = len(lines)
    val_select = random.sample(range(total_lines), int(validation_part * total_lines))
    val_lines = [lines[i] for i in range(total_lines) if i in val_select]
    train_lines = [lines[i] for i in range(total_lines) if i not in val_select]
    val = "\n".join(val_lines)
    train = "\n".join(train_lines)
    basename, dirname = os.path.basename(text_file), os.path.dirname(text_file)
    with open(os.path.join(dirname, 'train_' + basename), 'w') as f:
        f.write(train)
    with open(os.path.join(dirname, 'val_' + basename), 'w') as f:
        f.write(val)


if __name__ == '__main__':
    file = sys.argv[1]
    if len(sys.argv) == 3:
        val_prob = float(sys.argv[2])
    else:
        val_prob = 0.1
    split_train_val(file, val_prob)
