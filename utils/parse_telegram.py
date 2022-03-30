"""
CODE BASED ON:
https://github.com/mar-muel/artificial-self-AMLD-2020/blob/master/2/utils.py
"""
import json
import logging
import os
import glob
import sys

import pandas as pd
import unicodedata
import re
from tqdm import tqdm
import numpy as np
import random
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.clean import clean_links, clean_emojis, clean_several_whitespaces
import string

logger = logging.getLogger(__file__)
CACHE_PATH = 'cached_input_task2.txt'

def remove_control_characters(s):
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    control_char_regex = r'[\r\n\t]+'
    # replace \t, \n and \r characters by a whitespace
    s = re.sub(control_char_regex, ' ', s)
    # replace HTML codes for new line characters
    s = s.replace('&#13;', '').replace('&#10;', '')
    # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")


def clean(text):
    filters = (clean_links, clean_several_whitespaces)
    for f in filters:
        text = f(text)
    if len(text) == 0:
        # text = '.'
        return text
    if text[-1] not in string.punctuation:
        text += '.'
    return text


def read_conversation_data(data_path):
    """Read conversational data from either Chatistics or other data sources (in JSON) and returns Dataframe"""
    f_path = None
    if data_path is None:
        # infer file name
        data_folder = 'data'
        input_files = glob.glob(os.path.join(data_folder, '*.json'))
        if len(input_files) == 0:
            raise Exception(f'No files found in {data_folder}')
        elif len(input_files) > 1:
            raise Exception(f'Multiple files found in {data_folder}. Specify file with chatistics_data_path argument.')
        f_path = input_files[0]
    elif data_path is not None and os.path.isfile(data_path):
        f_path = data_path
    else:
        raise FileNotFoundError(f'Input data {data_path} could not be found')
    df = pd.read_json(f_path, encoding='utf-8')
    return df

def generate_input_task2(data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>'):
    """Generate input data for task 2"""
    # read conversation data
    df = read_conversation_data(data_path)
    # group messages by sender and generate output text file
    min_num_interactions_per_conversation = 10
    num_interactions = 0
    prev_sender_tag = None
    output = ''
    for conversation_name, g in tqdm(df.groupby('conversationWithName'), total=len(df['conversationWithName'].unique())):
        # only consider conversations between 2 people
        if len(g['senderName'].unique()) == 2 and len(g) > min_num_interactions_per_conversation:
            # sort by time
            g = g.sort_values('timestamp', ascending=True)
            for i, row in g.iterrows():
                sender_tag = speaker1_tag if row.outgoing else speaker2_tag
                if prev_sender_tag is None:
                    # beginning of chat with person
                    prev_sender_tag = sender_tag
                    current_messages = [remove_control_characters(row.text)]
                    continue
                if prev_sender_tag == sender_tag:
                    # concatenate/group messages by the same sender
                    current_messages.append(remove_control_characters(row.text))
                else:
                    # dump previous messsages
                    output += '{} {}\n'.format(prev_sender_tag, ' '.join(current_messages))
                    num_interactions += 1
                    # new response by other
                    prev_sender_tag = sender_tag
                    current_messages = [remove_control_characters(row.text)]
            if len(current_messages) > 0:
                output += '{} {}\n'.format(prev_sender_tag, ' '.join(current_messages))
                num_interactions += 1
    # write output data
    logger.info(f'Writing input file with {num_interactions:,} interactions to {CACHE_PATH}...')
    with open(CACHE_PATH, 'w') as f:
        f.write(output)

def get_input_task2(data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>', use_cache=True):
    """Load input data for task 2"""
    if not os.path.isfile(CACHE_PATH) or not use_cache:
        generate_input_task2(data_path, speaker1_tag='<speaker1>', speaker2_tag='<speaker2>')
    logger.info(f'Reading cached input file from {CACHE_PATH}...')
    with open(CACHE_PATH, 'r') as f:
        output = f.read()
    return output


def convert_chat_to_text(path, out_path, speakers_2):
    SPECIAL_TOKENS = {
        'speaker_1': '<speaker1>',
        'speaker_2': '<speaker2>',
    }

    with open(path, 'r') as f:
        data = json.load(f)

    speaker_1_id = str(data['personal_information']['user_id'])

    def get_chat_by_name(speaker_name):
        chats = data['chats']['list']
        for chat in chats:
            if speaker_name == str(chat['name']).lower():
                return chat
        return None

    def get_speaker_tag(from_id, speaker_2_id):
        from_id = str(from_id).lower()
        if speaker_1_id in from_id:
            return 'speaker_1'
        elif speaker_2_id in from_id:
            return 'speaker_2'
        else:
            return 'another'

    def filter_links(text):
        result = ''
        if isinstance(text, str):
            result = text
        elif isinstance(text, list):
            for element in text:
                if isinstance(element, str):
                    result += element
                elif isinstance(element, dict):
                    result += element.get('text', '')
                else:
                    result += ''
        return result

    chats = dict()
    for speaker in speakers_2:
        speaker = str(speaker).lower()
        chats[speaker] = get_chat_by_name(speaker)

    output = ''

    for speaker, chat in chats.items():
        speaker_2_id = str(chat['id'])
        current_messages = []
        prev_sender_tag = None
        for message in tqdm(chat['messages'], total=len(chat['messages'])):
            from_id = message.get('from_id', None)
            if from_id is None:
                continue
            text = filter_links(message['text'])
            if len(text) == 0:
                continue
            sender_tag = get_speaker_tag(from_id, speaker_2_id)
            if prev_sender_tag is None:
                # beginning of chat with person
                prev_sender_tag = sender_tag
                current_messages = [clean(remove_control_characters(text))]
                continue
            if prev_sender_tag in SPECIAL_TOKENS.keys():
                if prev_sender_tag == sender_tag:
                    # concatenate/group messages by the same sender
                    current_messages.append(clean(remove_control_characters(text)))
                else:
                    # dump previous messsages
                    output += '{} {}\n'.format(SPECIAL_TOKENS[prev_sender_tag], ' '.join(current_messages))
                    # new response by other
                    prev_sender_tag = sender_tag
                    current_messages = [clean(remove_control_characters(text))]
        if len(current_messages) > 0 and (prev_sender_tag in SPECIAL_TOKENS.keys()):
            output += '{} {}\n'.format(SPECIAL_TOKENS[prev_sender_tag], ' '.join(current_messages))

    with open(out_path, 'w') as f:
        f.write(output)


def convert_channel_to_text(path, out_path, speakers_2=None):
    SPECIAL_TOKENS = {
        'speaker_1': '<speaker1>',
        'speaker_2': '<speaker2>',
    }

    with open(path, 'r') as f:
        chat = json.load(f)

    speaker_1_id = '<empty>'  #str(data['personal_information']['user_id'])

    def get_speaker_tag(from_id, speaker_2_id):
        from_id = str(from_id).lower()
        if speaker_1_id in from_id:
            return 'speaker_1'
        elif speaker_2_id in from_id:
            return 'speaker_2'
        else:
            return 'another'

    def filter_links(text):
        SPECIAL_REMOVE = ('@new_militarycolumnist', )
        result = ''
        if isinstance(text, str):
            result = text
        elif isinstance(text, list):
            for element in text:
                if isinstance(element, str):
                    result += element
                elif isinstance(element, dict):
                    result += element.get('text', '')
                else:
                    result += ''
        for sr in SPECIAL_REMOVE:
            result = result.replace(sr, '')
        return result

    def read_message(message):
        if isinstance(message, list):
            result = ''
            for m in message:
                result += read_message(m) + ' '
            result = result[:-1]
        elif isinstance(message, str):
            result = message
        elif isinstance(message, dict):
            result = message['text']
        else:
            result = ''
        return result

    def parse_date(d):
        from dateutil import parser
        return parser.parse(d)

    output = ''

    start_date = parse_date('2022-02-01T00:00:00')

    speaker_2_id = 'channel' + str(chat['id'])
    for message in tqdm(chat['messages'], total=len(chat['messages'])):
        from_id = message.get('from_id', None)
        date = parse_date(message['date'])
        if date < start_date:
            continue
        if from_id is None:
            continue
        text = read_message(message['text'])
        text = filter_links(text)
        if len(text) == 0:
            continue
        sender_tag = get_speaker_tag(from_id, speaker_2_id)
        if sender_tag != 'speaker_2':
            continue
        text_to_add = clean(remove_control_characters(text))
        if len(text_to_add) == 0:
            continue
        text_to_add = ' '.join([SPECIAL_TOKENS[sender_tag], text_to_add + '\n'])
        output += text_to_add

    with open(out_path, 'w') as f:
        f.write(output)



if __name__ == '__main__':
    data_path = sys.argv[1]
    speakers = sys.argv[2]
    speakers = speakers.split('_')
    out_path = os.path.join(os.path.dirname(data_path), 'result_tg_cleaned.txt')
    #convert_chat_to_text(data_path, out_path, speakers)
    convert_channel_to_text(data_path, out_path)