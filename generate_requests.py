import sys
import torch
from transformers import pipeline, set_seed


def get_all_prefixes(entity):
    entity = entity.strip()
    capitalized_entity = entity.capitalize()

    pairs = [
        {
            'prefix': f'What does {entity} do? {capitalized_entity}',
            'request_prefix': capitalized_entity
        },
        {
            'prefix': f'What is the duty of {entity}? {capitalized_entity}',
            'request_prefix': capitalized_entity
        },
        {
            'prefix': f'What is the purpose of {entity}? {capitalized_entity}',
            'request_prefix': capitalized_entity
        },
        {
            'prefix': f'Where can {entity} appear? {capitalized_entity} can appear',
            'request_prefix': capitalized_entity
        },
        {
            'prefix': f'Where can I find {entity}? Usually you can find {entity}',
            'request_prefix': capitalized_entity
        },
        {
            'prefix': f'Where can I see {entity}? Usually you can see {entity}',
            'request_prefix': capitalized_entity,
        },
        {
            'prefix': f'Generate at least three related search terms to {entity}.',
            'request_prefix': '',
        }
    ]
    return pairs


def prepare_caption(sentence, prefix, max_len=30):
    sentence = sentence.strip()
    if sentence.startswith(prefix['prefix']):
        sentence = sentence[len(prefix['prefix']):]
    sentence = sentence.split('.')[0]
    sentence = sentence.split('\n')[0]
    sentence = sentence.strip()
    sentence = ' '.join(sentence.split(' ')[:max_len])
    sentence = ' '.join([prefix['request_prefix'], sentence]).strip()
    return sentence


def sample(generator, prefix, seed=42):
    set_seed(seed)
    result = generator(prefix)
    return [p['generated_text'] for p in result]


@torch.no_grad()
def llm_generate(model_path):
    generator = pipeline('text-generation', model=model_path, do_sample=True, num_return_sequences=5, device=-1)
    for p in get_all_prefixes("exit sign"):
        res = sample(generator, p['prefix'])
        print('PREFIX', p)
        print('*********')
        all_caps = list()
        for pred in res:
            try:
                caption = prepare_caption(pred, p)
                all_caps.append(caption)
            except:
                continue
        print('CAPTIONS\n', '\n'.join(all_caps))


@torch.no_grad()
def instruct_generate(model_path):
    generator = pipeline('text-generation', model=model_path, do_sample=True, num_return_sequences=5, device=-1)
    res = sample(generator, 'Explain in a sentence in English where does rhino appear. Rhino usually appears')
    print(res)


if __name__ == "__main__":
    instruct_generate(sys.argv[1])
