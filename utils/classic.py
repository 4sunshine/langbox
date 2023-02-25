import os
import string
import sys
import re
import pandas as pd
import tqdm
import numpy as np
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from collections import defaultdict
import spacy


def get_target(term='ru'):
    ru_keywords = {"россия", "россию", "россии", "российск", "россией", "рф", "русски", "русско"}
    ua_keywords = {"украина", "украину", "украины", "украинск", "украиной", "укро", "украинск", "украинц", "украи"}
    us_keywords = {"сша", "америк", "штаты америки", "штатов америк"}
    target_keywords = {'', }
    exclude = {'', }
    if term == 'ru':
        target_keywords = ru_keywords
        exclude = ua_keywords | us_keywords
    elif term == 'us':
        target_keywords = us_keywords
        exclude = ru_keywords | ua_keywords
    elif term == 'ua':
        target_keywords = ua_keywords
        exclude = ru_keywords | us_keywords
    return target_keywords, exclude


TARGET_KEYWORDS, ADD_TO_EXCLUSION = get_target('ru')


def remove_substring(text, substring="<speaker2> "):
    with open(text, 'r') as f:
        all_text = f.read()
    rex = re.compile(rf"{substring}")
    all_text = rex.sub(r"", all_text)
    print(all_text[:100])
    with open(text[:-4] + '_corrected.txt', "w") as f:
        f.write(all_text)


def substitute_with_whitespace(text, substring="\\xa0"):
    with open(text, 'r') as f:
        all_text = f.read()
    rex = re.compile(rf"{substring}")
    all_text = rex.sub(r" ", all_text)
    print(all_text[:100])
    with open(text[:-4] + '_corrected.txt', "w") as f:
        f.write(all_text)


def multiple_keywords_filter(text):
    l_text = text.lower()
    return any([tw in l_text for tw in TARGET_KEYWORDS]) and not any([tw in l_text for tw in ADD_TO_EXCLUSION])


def read_russian_sentiment_dataset(path_to_csv):
    """https://github.com/dkulagin/kartaslov/tree/master/dataset/kartaslovsent"""
    """NEUT, PSTV, NGTV"""
    def encode_tag(tag):
        code = 0.
        if tag == "PSTV":
            code = 1.
        elif tag == "NGTV":
            code = -1.
        return code
    df = pd.read_csv(path_to_csv, delimiter=";")
    print(df.head())
    terms = df["term"].values.tolist()
    tags = df["tag"].values.tolist()
    values = df["value"].values.tolist()
    term2val = {t: v for t, v in zip(terms, values)}
    term2tag = {t: v for t, v in zip(terms, tags)}
    term2encoded = {t: encode_tag(v) for t, v in zip(terms, tags)}
    return term2tag, term2val, term2encoded


def calculate_mean_sentiment(frequencies, term2encoded, stat_level=0.8):
    freqs_np = np.array([frequencies[k] for k in frequencies])
    quantile = np.quantile(freqs_np, stat_level)
    total_occurrence = 0
    weighted_sentiment = 0

    for k, freq in frequencies.items():
        if freq > quantile:
            total_occurrence += freq
            weighted_sentiment += term2encoded.get(k, 0.) * freq

    if total_occurrence > 0:
        result = weighted_sentiment / total_occurrence
    else:
        print("Cannot calculate mean sentiment")
        result = 0.
    return result


def calculate_sentence_sentiment(lemmas, term2encoded, term2val):
    result = 0
    value = 0
    has_negative = False
    for lemma in lemmas:
        cur_sentiment = term2encoded.get(lemma, 0)
        has_negative |= cur_sentiment < 0
        result += cur_sentiment
        value += term2val.get(lemma, 0)
    if len(lemmas) > 0:
        value /= len(lemmas)
        if value >= 0.55 / len(lemmas):
            value = 1
        elif value <= -0.35 / len(lemmas):
            value = -1
        else:
            value = 0
    return result, value, has_negative


def spacy_filter(text, model):
    doc = model(text)
    exclusions = tuple()
    remove_lemmas = ("год", "объявить", "заявить", "назвать", "рассказать")
    skip = ("-",)
    target_pos = {'NOUN', 'PROPN', "VERB", "ADJ"}
    result_lemmas = list()
    for token in doc:
        lowered_token = token.text.lower()
        if lowered_token in skip:
            continue
        if lowered_token in TARGET_KEYWORDS:
            continue
        if any([keyword in lowered_token for keyword in TARGET_KEYWORDS | ADD_TO_EXCLUSION]):
            continue
        if token.lemma_ in remove_lemmas:
            continue
        if token.text in exclusions:
            result_lemmas.append(token.lemma_)
        elif token.pos_ in target_pos:
            result_lemmas.append(token.lemma_)
    return result_lemmas


def gather_line_stats(text_file, russian_sentiment_dict_path):
    with open(text_file, 'r') as f:
        all_sents = [l.strip() for l in f.readlines()]
    target_utts = filter(multiple_keywords_filter, all_sents)
    result = list(target_utts)
    language_model = spacy.load('ru_core_news_md')
    term2tag, term2val, term2encoded = read_russian_sentiment_dataset(russian_sentiment_dict_path)

    sentiment = defaultdict(int)
    frequencies = defaultdict(int)
    soft_sentiment = defaultdict(int)
    negative_words = defaultdict(int)
    for news in tqdm.tqdm(result):
        lemmas = spacy_filter(news, language_model)
        sent_sentiment, sent_value, has_negative_word = calculate_sentence_sentiment(lemmas, term2encoded, term2val)

        if has_negative_word:
            negative_words['with_negative'] += 1
        else:
            negative_words['without_negative'] += 1

        if sent_sentiment > 0:
            sentiment['positive'] += 1
        elif sent_sentiment < 0:
            sentiment['negative'] += 1
        else:
            sentiment['neutral'] += 1

        if sent_value > 0:
            soft_sentiment['positive'] += 1
        elif sent_value < 0:
            soft_sentiment['negative'] += 1
        else:
            soft_sentiment['neutral'] += 1

        for l in lemmas:
            frequencies[l] += 1

    frequencies = sorted(frequencies.items(), key=lambda x: x[1])
    frequencies = {k: float(v) for k, v in frequencies}

    mean_sentiment = calculate_mean_sentiment(frequencies, term2encoded)
    print(f'mean_sentiment {mean_sentiment}')
    print(f'sentiment {sentiment}')
    print(f'soft sentiment {soft_sentiment}')
    print(f'at least one negative {negative_words}')

    #wordcloud = WordCloud().generate_from_frequencies(frequencies)
    #image = wordcloud.to_image()
    #image.show()
    #image.save("wordcloud.png")


if __name__ == "__main__":
    gather_line_stats(sys.argv[1], sys.argv[2])
    exit(0)
    #remove_substring(sys.argv[1])
    substitute_with_whitespace(sys.argv[1])

