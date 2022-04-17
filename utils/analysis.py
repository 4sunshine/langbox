import spacy
from spacy import displacy


def pos_tagger_spacy(text, nlp_model, max_single_len=3, max_pieces=3):
    document = nlp_model(text)
    all_subtrees = dict()
    all_childs = dict()
    all_anchestors = dict()
    verb_child = dict()
    for token in document:
        #print(token.text)
        #print(token.pos_)
        #print(dir(token))
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
    print('SUBTREES', all_subtrees)
    print('CHILD', all_childs)
    print('ANCHESTORS', all_anchestors)
    flat_anchestors = [a for k, v in all_anchestors.items() for a in v]
    print(flat_anchestors)
    result_names = []
    print('VERB_CHILD', verb_child)
    another_names = []
    for verb, v_children in verb_child.items():
        for child in v_children:
            if str(child) in all_subtrees.keys():
                another_names.append(str(child))

    for name, subtree in all_subtrees.items():
        child = all_childs[name]
        # print(name)
        # print(subtree)
        # print(child + [name])
        if set(subtree) == set(child + [name]):
            if name in flat_anchestors:
                result_names.append(name)
    result_string = ' и '.join([' '.join(all_subtrees[name][:max_single_len]) for name in result_names[:max_pieces]])
    result_string = ' '.join(['Фотография', result_string])

    another_string = ' и '.join([' '.join(all_subtrees[name][:max_single_len]) for name in another_names[:max_pieces]])
    another_string = ' '.join(['Фотография', another_string])

    print(result_string)
    print('ANOTHER', another_string)
    print(text)


if __name__ == '__main__':
    texts = ['Вооружённые Силы РФ приступили к перевозке главных кошек на тележках',
             'Президент Владимир Путин начал и продолжил перевозку главных кошек на тележках',
             'Президент Владимир Путин встретился с Владимиром Зеленским и с Рамзаном',
             'Ученые давно знают, что некоторые вирусы из группы COVID-19 провоцируют иммунные клетки на рост опухолей, но пока что для этого не достаточно специфических симптомов. В новой работе американские исследователи с помощью томографии и компьютерного зрения показали, что одна из причин развития рака легких у ВИЧ-инфицированных — вовлечение бактериальной РНК.',
             'Инженеры создали программное обеспечение, которое делает инструменты, которые помогают создавать детекторы лжи и снимать их на камеру. Самое главное, что оно может записывать с видеокамеры на компьютере такие действия, и только потом можно использовать их на практике.']
    nlp = spacy.load('ru_core_news_md')
    for text in texts:
        text = text.split('.')[0]
        pos_tagger_spacy(text, nlp)
