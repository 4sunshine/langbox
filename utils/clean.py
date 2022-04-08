import os.path
import re
import sys


def clean_links(text):
    cleaned = re.sub(r"\S*https?:\S*", '', text)
    return cleaned


def clean_several_whitespaces(text):
    text = text.replace(" . ", " ")
    text = text.replace(" .", " ")
    cleaned = re.sub(" +", " ", text)
    cleaned = cleaned.replace(" \n", ".\n")
    cleaned = cleaned.replace("..\n", ".\n")
    return cleaned


def clean_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ':) ', text)


if __name__ == '__main__':
    file = sys.argv[1]
    with open(file, 'r') as f:
        text = f.read()
    basename = os.path.basename(file)
    name, _ = os.path.splitext(basename)
    dirname = os.path.dirname(file)
    cleaned_text = clean_links(text)
    cleaned_text = clean_emojis(cleaned_text)
    cleaned_text = clean_several_whitespaces(cleaned_text)
    new_name = os.path.join(dirname, name + '_cleaned.txt')
    with open(new_name, 'w') as fw:
        fw.write(cleaned_text)
