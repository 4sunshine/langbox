import sys
import os
import string
import spacy
import re
import torch.cuda
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rutermextract import TermExtractor


class RuParaphraser:
    """https://huggingface.co/cointegrated/rut5-base-paraphraser"""
    def __init__(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def paraphrase(self, text, beams=5, grams=4, do_sample=False):
        x = self.tokenizer(text, return_tensors='pt', padding=True).to(self.model.device)
        max_size = int(x.input_ids.shape[1] * 0.5 + 10)
        out = self.model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams,
                                  max_length=max_size, do_sample=do_sample)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


def pos_tagger_spacy(text, nlp_model, max_single_len=3, topk=2, paraphraser=None):
    document = nlp_model(text)
    all_subtrees = dict()
    all_childs = dict()
    all_anchestors = dict()
    verb_child = dict()
    for token in document:
        text = token.text.translate(str.maketrans('', '', string.punctuation))
        if len(text) == 0:
            continue
        pos = token.pos_
        if pos in ('NOUN', 'PROPN'):
            subtree = [str(s) for s in token.subtree]
            all_subtrees[token.text] = subtree
            children = [str(c) for c in token.children]
            all_childs[token.text] = children
            anchestors = [str(a) for a in token.ancestors]
            all_anchestors[token.text] = anchestors
        if pos in ('VERB',):
            verb_child[token.text] = [str(c) for c in token.children]

    another_names = []
    for verb, v_children in verb_child.items():
        for child in v_children:
            if str(child) in all_subtrees.keys():
                another_names.append(str(child))

    if len(another_names) == 0:
        another_names = [k for k in all_anchestors.keys()][:topk]

    another_string = ' и '.join([' '.join(all_subtrees[name][:max_single_len]) for name in another_names[:topk]])
    # another_string = ' '.join(['Фотография', another_string.lower()])

    another_string = re.sub(r'\s+([?.,;!"])', r'\1', another_string)

    if paraphraser is not None:
        another_string = paraphraser.paraphrase(another_string)
        another_string = re.sub(r'\s+([?.,;!"])', r'\1', another_string)

    return another_string


def prepare_gen_texts(text_file, paraphraser=None):
    with open(text_file, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
    nlp = spacy.load('ru_core_news_md')
    if paraphraser is not None:
        paraphraser = RuParaphraser(paraphraser)
    output = ''
    print('Extracting main tags')
    print('* * * * *')
    for text in texts:
        text = text.split('.')[0]
        result = pos_tagger_spacy(text, nlp, paraphraser=paraphraser)
        output += result + '\n'
        print(result)
        print('***')
    dirname = os.path.dirname(text_file)
    basename = os.path.basename(text_file)
    target_path = os.path.join(dirname, 'generation_' + basename)
    with open(target_path, 'w') as f:
        f.write(output)
    return target_path


def prepare_gen_texts_key(text_file, paraphraser=None):
    with open(text_file, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
    nlp = TermExtractor()
    if paraphraser is not None:
        paraphraser = RuParaphraser(paraphraser)
    output = ''
    print('Extracting main tags')
    print('* * * * *')
    for text in texts:
        result = rukeyextract(text, nlp, paraphraser=paraphraser)
        output += result + '\n'
        print(result)
        print('***')
    dirname = os.path.dirname(text_file)
    basename = os.path.basename(text_file)
    target_path = os.path.join(dirname, 'generation_' + basename)
    with open(target_path, 'w') as f:
        f.write(output)
    return target_path


def rukeyextract(text, nlp_model, topk=2, paraphraser=None):
    top_words = [term.normalized for term in nlp_model(text)[:topk]]
    result = []
    if paraphraser is not None:
        for key_words in top_words:
            num_words = len(key_words.split(' '))
            if num_words > 1:
                key_words = paraphraser.paraphrase(key_words)
                key_words = ' '.join(key_words.split(' ')[:num_words])
            result.append(key_words)
    else:
        result = top_words
    result_string = ' и '.join(result)
    result_string = re.sub(r"[()\"#/@;:<>{}`+=~|.!?,]", "", result_string)  # REMOVE PUNCTUATION WITHOUT '-'
    return result_string


if __name__ == '__main__':
    text_file = sys.argv[1]
    paraphraser = sys.argv[2]
    prepare_gen_texts_key(text_file, paraphraser=paraphraser)
