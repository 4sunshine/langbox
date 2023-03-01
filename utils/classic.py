import json
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
import random


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.clean import clean_emojis, clean_several_whitespaces


def get_target(term='ru', opt_exclude=None):
    ru_keywords = {"россия", "россию", "россии", "российск", "россией", "рф", "русски", "русско", "российский"}
    ua_keywords = {"украина", "украину", "украины", "украинск", "украиной", "укро", "украинск", "украинц", "украи", "украинский"}
    us_keywords = {"сша", "америк", "штаты америки", "штатов америк"}
    target_keywords = {'', }
    exclude = set()
    if term == 'ru':
        target_keywords = ru_keywords
        if not opt_exclude:
            exclude = ua_keywords | us_keywords
    elif term == 'us':
        target_keywords = us_keywords
        if not opt_exclude:
            exclude = ru_keywords | ua_keywords
    elif term == 'ua':
        target_keywords = ua_keywords
        if not opt_exclude:
            exclude = ru_keywords | us_keywords
    return target_keywords, exclude


def remove_substring(text, substring="<speaker2> ", additional_clean=True):
    with open(text, 'r') as f:
        all_text = f.read()
    rex = re.compile(rf"{substring}")
    all_text = rex.sub(r"", all_text)
    if additional_clean:
        all_text = clean_emojis(all_text, substitute_with='. ')
        all_text = clean_several_whitespaces(all_text)
    print(all_text[:100])
    target_path = text[:-4] + '_corrected.txt'
    with open(target_path, "w") as f:
        f.write(all_text)
    return target_path


def substitute_with_whitespace(text, substring="\\xa0"):
    with open(text, 'r') as f:
        all_text = f.read()
    rex = re.compile(rf"{substring}")
    all_text = rex.sub(r" ", all_text)
    print(all_text[:100])
    target_path = text[:-4] + '_corrected.txt'
    with open(target_path, "w") as f:
        f.write(all_text)
    return target_path


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
    # print(df.head())
    terms = df["term"].values.tolist()
    tags = df["tag"].values.tolist()
    values = df["value"].values.tolist()
    term2val = {t: v for t, v in zip(terms, values)}
    term2tag = {t: v for t, v in zip(terms, tags)}
    term2encoded = {t: encode_tag(v) for t, v in zip(terms, tags)}
    keys_to_remove = {'украина', }  # Ukraine presented in the dataset while other countries not
    for k in keys_to_remove:
        del term2encoded[k]
        del term2tag[k]
        del term2val[k]
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
    has_positive = False
    extra_negative = False
    extra_positive = False
    for lemma in lemmas:
        cur_sentiment = term2encoded.get(lemma, 0)
        has_negative |= cur_sentiment < 0
        has_positive |= cur_sentiment > 0
        result += cur_sentiment
        emotion_val = term2val.get(lemma, 0)
        value += emotion_val
        extra_negative |= emotion_val < -0.95
        extra_positive |= emotion_val > 0.95
    if len(lemmas) > 0:
        value /= len(lemmas)
        if value >= 0.55 / len(lemmas):
            value = 1
        elif value <= -0.35 / len(lemmas):
            value = -1
        else:
            value = 0

    wide_emotion = extra_negative and extra_positive
    extra_negative = extra_negative and not wide_emotion
    extra_positive = extra_positive and not wide_emotion
    return result, value, has_negative, has_positive, extra_negative, extra_positive, wide_emotion


def spacy_filter(text, model):
    doc = model(text)
    exclusions = tuple()
    remove_lemmas = ("год", "объявить", "заявить", "назвать", "рассказать", "новость", "риа", "сообщить", "быть", "стать",
                     "сказать", "рассказать", "мочь", "сообщать", "медуза", "писать")
    skip = ("-",)
    target_pos = {'NOUN', 'PROPN', "VERB", "ADJ"}
    result_lemmas = list()
    for i, token in enumerate(doc):
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
            result_lemmas.append((token.lemma_, i))
        elif token.pos_ in target_pos:
            result_lemmas.append((token.lemma_, i))
    return result_lemmas


def gather_line_stats(text_file, russian_sentiment_dict_path):
    with open(text_file, 'r') as f:
        target_utts = [l.strip() for l in f.readlines()]
    if FILTER_KEYWORDS:
        target_utts = filter(multiple_keywords_filter, target_utts)
    result = list(target_utts)
    language_model = spacy.load('ru_core_news_md')
    term2tag, term2val, term2encoded = read_russian_sentiment_dataset(russian_sentiment_dict_path)

    sentiment = defaultdict(int)
    frequencies = defaultdict(int)
    soft_sentiment = defaultdict(int)
    negative_words = defaultdict(int)
    all_lemmas = list()
    for news in tqdm.tqdm(result):
        lemmas = spacy_filter(news, language_model)
        all_lemmas.append(lemmas)

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
    print(f'Most frequent 100:', frequencies[-100:])
    frequencies = {k: float(v) for k, v in frequencies}

    mean_sentiment = calculate_mean_sentiment(frequencies, term2encoded)
    print(f'mean_sentiment {mean_sentiment}')
    print(f'sentiment {sentiment}')
    print(f'soft sentiment {soft_sentiment}')
    print(f'at least one negative {negative_words}')

    result_data = {
        'lemmas': all_lemmas,
        'mean_sentiment': mean_sentiment,
        'sentiment': sentiment,
        'soft_sentiment': soft_sentiment,
        'with_one_negative': negative_words,
    }

    with open(text_file[:-4] + '_dataset.json', 'w') as f:
        json.dump(result_data, f)

    wordcloud = WordCloud(max_words=200).generate_from_frequencies(frequencies)
    image = wordcloud.to_image()
    image.show()
    image.save("wordcloud.png")


def analyze_lemmas_dataset(ds, russian_sentiment_dict_path):
    print('######')
    dataset_name = os.path.splitext(ds)[0]
    print('Analyzing dataset', dataset_name)
    with open(ds, 'r') as f:
        data = json.load(f)
    lemmas = data['lemmas']
    num_lemmas = np.array([len(ls) for ls in lemmas])
    position2num = np.array([ls[-1][-1] / len(ls) for ls in lemmas if len(ls)])

    def filter_keywords(lm):
        text = ' '.join([l[0] for l in lm])
        if not OPT_EXCLUDE:
            return any([tw in text for tw in TARGET_KEYWORDS]) and not any([tw in text for tw in ADD_TO_EXCLUSION])
        else:
            return any([tw in text for tw in TARGET_KEYWORDS])

    if TARGET_KEYWORDS:
        lemmas = list(filter(filter_keywords, lemmas))

    # print(f'Mean lemmas', np.mean(num_lemmas))
    # print(f'Max lemmas', np.max(num_lemmas))
    # print(f'Std lemmas', np.std(num_lemmas))
    # print(f'Mean pos/len', np.mean(position2num))
    # print(f'Max pos/len', np.max(position2num))
    # print(f'Std pos/len', np.std(position2num))

    term2tag, term2val, term2encoded = read_russian_sentiment_dataset(russian_sentiment_dict_path)

    frequencies = defaultdict(int)
    attention_freq = defaultdict(float)
    positive_items = defaultdict(int)
    negative_items = defaultdict(int)

    sentiment = defaultdict(int)
    soft_sentiment = defaultdict(int)
    negative_words = defaultdict(int)
    positive_words = defaultdict(int)
    extra_emotion = defaultdict(int)

    full_exclusion = TARGET_KEYWORDS | ADD_TO_EXCLUSION
    additional_exclusion = ("видео", "день",)

    for sentence in tqdm.tqdm(lemmas):
        result_lemmas = list()
        for lemma, position in sentence:
            if position > MAX_LEMMA_POSITION:
                break
            if lemma in TARGET_KEYWORDS or lemma in additional_exclusion:
                continue
            if any([(keyword in lemma) or (lemma in keyword) for keyword in full_exclusion]):
                continue
            frequencies[lemma] += 1
            attention_freq[lemma] += (2 - (position / MAX_LEMMA_POSITION) ** 2)
            result_lemmas.append(lemma)

        sent_sentiment, sent_value, has_negative, has_positive, extra_negative, extra_positive, wide_emotion =\
            calculate_sentence_sentiment(result_lemmas, term2encoded, term2val)

        if has_negative:
            negative_words['with_negative'] += 1
        else:
            negative_words['without_negative'] += 1

        if has_positive:
            positive_words['with_positive'] += 1
        else:
            positive_words['without_positive'] += 1

        if sent_sentiment > 0:
            sentiment['positive'] += 1
            for lemma in result_lemmas:
                if int(term2encoded.get(lemma, 0)) == 0:
                    positive_items[lemma] += 1
        elif sent_sentiment < 0:
            sentiment['negative'] += 1
            for lemma in result_lemmas:
                if int(term2encoded.get(lemma, 0)) == 0:
                    negative_items[lemma] += 1
        else:
            sentiment['neutral'] += 1

        if sent_value > 0:
            soft_sentiment['positive'] += 1
        elif sent_value < 0:
            soft_sentiment['negative'] += 1
        else:
            soft_sentiment['neutral'] += 1

        if extra_positive:
            extra_emotion['extra_positive'] += 1
        elif extra_negative:
            extra_emotion['extra_negative'] += 1
        elif wide_emotion:
            extra_emotion['wide'] += 1
        else:
            extra_emotion['no_extra_emotion'] += 1

    frequencies = sorted(frequencies.items(), key=lambda x: x[1])
    #print(f'Most frequent 100:', frequencies[-100:])
    frequencies = {k: float(v) for k, v in frequencies}

    attention_freq = sorted(attention_freq.items(), key=lambda x: x[1])
    #print(f'Most attention frequent 100:', attention_freq[-100:])
    attention_freq = {k: float(v) for k, v in attention_freq}

    if len(frequencies.keys()) > 0:
        mean_sentiment = calculate_mean_sentiment(frequencies, term2encoded)
    else:
        mean_sentiment = 0

    def relative_dict(source_dict):
        sum_vals = sum([source_dict[k] for k in source_dict])
        relative = {k + '_part': round(100 * source_dict[k] / sum_vals, 2) for k in sorted(source_dict)}
        full = {k: source_dict[k] for k in sorted(source_dict)}
        return {**full, **relative}

    sentiment = relative_dict(dict(sentiment))
    soft_sentiment = relative_dict(dict(soft_sentiment))
    negative_words = relative_dict(dict(negative_words))
    positive_words = relative_dict(dict(positive_words))
    extra_emotion = relative_dict(dict(extra_emotion))

    print(f'mean_sentiment {mean_sentiment}')
    # print(f'sentiment: {sentiment}')
    # print(f'soft sentiment: {soft_sentiment}')
    # print(f'at least one negative {negative_words}')
    # print(f'at least one positive {positive_words}')
    # print(f'extra emotion distribution {extra_emotion}')

    result = {
        'mean_sentiment': mean_sentiment,
        'sentiment': sentiment,
        'soft_sentiment': soft_sentiment,
        'with_one_negative': negative_words,
        'with_one_positive': positive_words,
        'strong_emotion': extra_emotion,
    }
    result = {**RESULT_DATA, **result}
    target_name = dataset_name + f"_analysis_{result['code']}_{result['max_lemma_position']}_" \
                                 f"{result['opt_exclude']}_{result['filter_keywords']}"
    # with open(target_name + ".json",
    #           'w') as f:
    #     json.dump(result, f)

    FULL_RESULT.append(result)


    def general_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if word not in term2encoded:
            return f"hsl(216, {random.randint(80, 100)}%, {random.randint(50, 80)}%)"
        encode = term2encoded[word]
        if encode == 0:
            return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)
        elif encode == -1:
            return f"hsl(350, {random.randint(80, 100)}%, {random.randint(50, 80)}%)"
        else:
            return f"hsl(149, {random.randint(80, 100)}%, {random.randint(50, 80)}%)"

    def negative_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return f"hsl({random.randint(0, 20)}, {random.randint(60, 95)}%, {random.randint(50, 80)}%)"  # 350

    def positive_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return f"hsl({random.randint(110, 153)}, {random.randint(60, 95)}%, {random.randint(50, 80)}%)"

    frequencies_sorted = {k: float(v) for k, v in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:MAX_WORDS]}

    if len(frequencies.keys()) == 0:
        return

    wordcloud = WordCloud(max_words=MAX_WORDS, width=1040, height=480).generate_from_frequencies(frequencies_sorted)
    wordcloud.recolor(color_func=general_color_func, random_state=3)

    image = wordcloud.to_image()

    image_dirname = os.path.join(os.path.dirname(ds), os.path.splitext(ds)[0] + "_wcloud")
    os.makedirs(image_dirname, exist_ok=True)
    image_basename = os.path.basename(target_name)
    image_path = os.path.join(image_dirname, image_basename + ".png")
    image.save(image_path)

    # wordcloud = WordCloud(max_words=300).generate_from_frequencies(attention_freq)
    # image = wordcloud.to_image()
    # # image.show()
    # image.save(target_name + "_wcloud_attend.png")

    negative_items = {k: float(v) for k, v in sorted(negative_items.items(), key=lambda x: x[1], reverse=True)[:MAX_WORDS]}

    wordcloud = WordCloud(max_words=MAX_WORDS, width=520, height=480).generate_from_frequencies(negative_items)
    wordcloud.recolor(color_func=negative_color_func, random_state=3)
    image = wordcloud.to_image()

    image_path = os.path.join(image_dirname, image_basename + "_negative.png")
    image.save(image_path)

    positive_items = {k: float(v) for k, v in sorted(positive_items.items(), key=lambda x: x[1], reverse=True)[:MAX_WORDS]}

    wordcloud = WordCloud(max_words=MAX_WORDS, width=520, height=480).generate_from_frequencies(positive_items)
    wordcloud.recolor(color_func=positive_color_func, random_state=3)
    image = wordcloud.to_image()

    image_path = os.path.join(image_dirname, image_basename + "_positive.png")
    image.save(image_path)


def get_start_date(*channels):
    for channel in channels:
        with open(channel, 'r') as f:
            data = json.load(f)
        def parse_date(d):
            from dateutil import parser
            return parser.parse(d)

        for message in data['messages']:
            date = parse_date(message['date'])
            break

        print(os.path.basename(channel), 'started at', date)


def analyse_result(result):
    with open(result, 'r') as f:
        data = json.load(f)
    refactor = list()
    for d in data:
        cur_data = dict()
        for k, v in d.items():
            if isinstance(v, dict):
                prefix = ''.join([w[0] for w in k.split('_')])
                for sk, sv in v.items():
                    cur_data['_'.join([prefix, sk])] = sv
            else:
                cur_data[k] = v
        refactor.append(cur_data)

    df = pd.DataFrame(refactor)
    df['mean_sentiment'] = df['mean_sentiment'].apply(lambda x: x * 100)
    # df.to_excel("full_analysis.xlsx")
    df['dataset'] = df['dataset'].apply(lambda x: os.path.basename(x))

    df_extra = df[
        ["dataset", "code", "filter_keywords", "opt_exclude", "se_extra_negative_part", "se_extra_positive_part"]]
    df_extra = df_extra.groupby(["dataset", "code", "filter_keywords", "opt_exclude"]).mean()
    df_extra = df_extra.round(1)

    df = df.groupby(["dataset", "code", "filter_keywords", "opt_exclude"]).mean()
    df = df.round(1)

    df.to_excel("analysis.xlsx")

    df_extra.to_excel("analysis_extra.xlsx")

    print(df.head())


if __name__ == "__main__":
    datasets = [
        "zl.json",
        "zn.json",
        "rianag.json",
        "tassag.json",
    ]
    FULL_RESULT = list()
    MAX_WORDS = 300

    for FILTER_KEYWORDS in (False, True):
        for CODE in ('ru', 'us', 'ua'):
            for OPT_EXCLUDE in (False, True):
                for MAX_LEMMA_POSITION in (20, 30, 40):
                    for ds in datasets:
                        TARGET_KEYWORDS, ADD_TO_EXCLUSION = get_target(CODE, OPT_EXCLUDE)
                        # MAX_LEMMA_POSITION = 40  # ~2x Twitter limit
                        if not FILTER_KEYWORDS:
                            TARGET_KEYWORDS, ADD_TO_EXCLUSION = set(), set()

                        RESULT_DATA = {
                            'code': CODE,
                            'target_keywords': list(TARGET_KEYWORDS),
                            'filter_keywords': FILTER_KEYWORDS,
                            'max_lemma_position': MAX_LEMMA_POSITION,
                            'exclusion': list(ADD_TO_EXCLUSION),
                            'opt_exclude': OPT_EXCLUDE,
                            'dataset': os.path.splitext(ds)[0],
                        }
                        analyze_lemmas_dataset(ds, sys.argv[2])

    analysis_path = 'full_analysis_result.json'

    with open(analysis_path, 'w') as f:
        json.dump(FULL_RESULT, f)

    analyse_result(analysis_path)

    # gather_line_stats(sys.argv[1], sys.argv[2])
    # exit(0)
    # remove_substring(sys.argv[1])
    # substitute_with_whitespace(sys.argv[1])

