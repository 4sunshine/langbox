"""
CODE BASED ON:
https://github.com/mar-muel/artificial-self-AMLD-2020/blob/master/2/utils.py
"""
import json
import os
import unicodedata
import re
import string
import argparse

from tqdm import tqdm

from clean import clean_links, clean_emojis, clean_several_whitespaces


SPECIAL_TOKENS = {
    'speaker_1': '<speaker1>',
    'speaker_2': '<speaker2>',
}


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


def convert_chat_to_text(path, out_path, speakers_2):
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
                    type = element.get('type', None)
                    cur_text = element.get('text', '')
                    if type != 'link':
                        result += cur_text #element.get('text', '')
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


def convert_channel_to_text(path, out_path, start_date='1980-01-01T00:00:00',
                            special_remove=()):
    with open(path, 'r') as f:
        chat = json.load(f)

    def get_speaker_tag(from_id, speaker_2_id):
        from_id = str(from_id).lower()
        if speaker_2_id in from_id:
            return 'speaker_2'
        else:
            return 'another'

    def filter_links(text, advertising=('#реклама', )):
        for sr in special_remove:
            text = text.replace(sr, '')
        for ad in advertising:
            if ad in text:
                text = ''
        return text

    def read_message(message):
        if isinstance(message, list):
            result = ''
            for m in message:
                result += read_message(m) + ' '
            result = result[:-1]
        elif isinstance(message, str):
            result = message
        elif isinstance(message, dict):
            text_type = message.get('type', None)
            if text_type != 'link':
                result = message.get('text', '')
            else:
                result = ''
        else:
            result = ''
        return result

    def parse_date(d):
        from dateutil import parser
        return parser.parse(d)

    output = ''

    start_date = parse_date(start_date)

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

    cleaned_text = clean_links(output)
    # cleaned_text = clean_emojis(cleaned_text)
    output = clean_several_whitespaces(cleaned_text)

    with open(out_path, 'w') as f:
        f.write(output)


def main():
    parser = argparse.ArgumentParser(description='Telegram parser')
    parser.add_argument('--data_path', type=str, help='Telegram data in JSON format', required=True)
    parser.add_argument('--speakers', type=str, help='Speakers to parse', default=None)
    parser.add_argument('--start_date', type=str, help='From which date parse', default='1980-01-01T00:00:00')
    parser.add_argument('--special_remove', type=str, help='Remove strings', default='Подробнее на N + 1')
    args = parser.parse_args()
    out_path = os.path.basename(args.data_path)
    basename = os.path.splitext(out_path)[0]
    out_path = os.path.join(os.path.dirname(args.data_path), basename + '_parsed.txt')
    if args.speakers is None:
        convert_channel_to_text(args.data_path, out_path,
                                start_date=args.start_date, special_remove=tuple(args.special_remove.split(';')))
    else:
        convert_chat_to_text(args.data_path, out_path, args.speakers)


if __name__ == '__main__':
    main()
