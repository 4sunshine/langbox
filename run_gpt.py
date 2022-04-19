import os
import sys

from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    get_linear_schedule_with_warmup, AutoModelWithLMHead, AutoTokenizer
)
import torch
from utils.sample import sample_sequence
from utils.analysis import prepare_gen_texts
from utils.web import image_download as downloader
from translate import Translator
import time


def sample(model, tokenizer, initial_text, max_history=3, max_generation_steps=10, downloader=None,
           save_dir='runs/gpt_result', translator=None):
    if downloader is not None:
        os.makedirs(save_dir, exist_ok=True)
    speaker2_tag = '<speaker2>'
    speaker2_tag_id = tokenizer.convert_tokens_to_ids(speaker2_tag)
    history = []
    for i in range(max_generation_steps):
        history_cut = min(len(history), max_history)
        recent_history = history[-history_cut:]
        # concatenate history into single string and add trigger word "bot:"
        if i > 0:
            history_str = '{}\n{}'.format('\n'.join(recent_history), speaker2_tag)
        else:
            history_str = f'{speaker2_tag} {initial_text}'
        # tokenize text and convert into vocabulary ids (input ids)
        history_enc = tokenizer.encode(history_str, add_special_tokens=True)

        # sample_sequence(conversation, model, num_samples=1, device='cuda', max_length=80, temperature=1.0, top_k=0, top_p=0.8)

        with torch.no_grad():
            out_ids = sample_sequence(history_enc, model)
        if i > 0:
            out_ids = out_ids[:, len(history_enc):].tolist()[0]
        else:
            out_ids = out_ids[:, 1:].tolist()[0]
        # Select part before speaker tags as answer
        j = 0
        for j, out_id in enumerate(out_ids):
            if out_id == speaker2_tag_id:
                break
        # answer = '{} {}'.format(speaker1_tag, tokenizer.decode(out_ids[:i]))
        decoded_string = tokenizer.decode(out_ids[:j])
        answer = '{} {}'.format(speaker2_tag, decoded_string)
        if (downloader is not None) and (i == 0):
            request = ' '.join(decoded_string.split(' ')[:6])
            if translator is not None:
                request = translator.translate(request)
            result = downloader(request, save_dir)
            if result is not None:
                basename = os.path.basename(result)
                target_name = os.path.splitext(basename)[0]
                target_path = os.path.join(save_dir, target_name + '.txt')
                with open(os.path.join(target_path), 'w') as f:
                    f.write(decoded_string + '\n')

        # print(answer)
        # add answer to history
        history.append(answer)
    result = '\n'.join(history)
    result = result.replace(speaker2_tag + ' ', '')
    return result


def infer_channel_gpt3(checkpoint_path, initial_strings_file):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    model = AutoModelWithLMHead.from_pretrained(checkpoint_path).cuda()
    model.eval()
    with open(initial_strings_file, 'r') as f:
        initial_strings = [line.strip() for line in f.readlines()]

    # translator = Translator('en', 'ru', 'mymemory')
    print('Begin sampling')
    output = ''
    for input_str in initial_strings:
        result = sample(model, tokenizer, input_str, max_generation_steps=1, downloader=None,
                        translator=None)
        output += result + '\n'
        print(result)
        print('***')

    dirname = os.path.dirname(initial_strings_file)
    basename = os.path.basename(initial_strings_file)
    target_path = os.path.join(dirname, 'predict_' + basename)
    with open(target_path, 'w') as f:
        f.write(output)
    return target_path


def gpt_dalle_prepare(checkpoint_path, initial_strings_file):
    gpt_predictions_file = infer_channel_gpt3(checkpoint_path, initial_strings_file)
    prepared_for_dalle_file = prepare_gen_texts(gpt_predictions_file)


if __name__ == '__main__':
    checkpoint = sys.argv[1]
    sampling_file = sys.argv[2]
    predict_path = infer_channel_gpt3(checkpoint, sampling_file)
