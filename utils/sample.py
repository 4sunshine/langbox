import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(conversation, model, num_samples=1, device='cuda', max_length=80,
                    temperature=1.0, top_k=0, top_p=0.8):
    """Generate next tokens from pervious conversation"""
    context = torch.tensor(conversation, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(max_length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)
            # scale by temperature
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            # filter by top-k/top-p
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def sample(model, tokenizer, sampling_history=None, max_history=2, no_info=True, max_generation_steps=10):
    speaker1_tag = '<speaker1>'
    speaker2_tag = '<speaker2>'
    speaker1_tag_id = tokenizer.convert_tokens_to_ids(speaker1_tag)
    speaker2_tag_id = tokenizer.convert_tokens_to_ids(speaker2_tag)
    # history = f"""
    # {speaker2_tag} Привет )
    # {speaker1_tag} Привет, чё как?
    # {speaker2_tag} Нормально, сам как?
    # {speaker1_tag} Хорошо
    # {speaker2_tag} Давай поговорим?
    # {speaker1_tag} О чём хочешь поговорить?"""
    # print(history)
    # print('\n[Chat with the model! Send "h" to see the full history]\n')
    # history = history.split('\n')
    if sampling_history is not None:
        with open(sampling_history, 'r') as f:
            history = f.readlines()
    else:
        history = [f"{speaker2_tag} Первое сообщение.\n",
                   f"{speaker2_tag} Второе сообщение.\n",
                   f"{speaker2_tag} Третье сообщение.\n",
                   f"{speaker2_tag} Четвёртое сообщение.\n"]

    for i in range(max_generation_steps):
        # message = None #'О стикерах вконтакте'
        # while not message:
        #     print(f'{speaker2_tag} writing...:')
        #     message = input()
        #     if message == 'h':
        #         print('\n'.join(history))
        #         message = None
        #     elif message == 'quit':
        #         break
        # # add new message to history
        # history.append(f'{speaker2_tag} {message}')
        # PERSONAL CHAT END
        # keep only most recent conversation as input to the model
        recent_history = history[-(2 * max_history):]
        # concatenate history into single string and add trigger word "bot:"
        history_str = '{}\n{}'.format('\n'.join(recent_history), speaker1_tag)
        history_str = '{}\n{}'.format('\n'.join(recent_history), speaker2_tag)
        # tokenize text and convert into vocabulary ids (input ids)
        history_enc = tokenizer.encode(history_str, add_special_tokens=True)
        with torch.no_grad():
            out_ids = sample_sequence(history_enc, model)
        out_ids = out_ids[:, len(history_enc):].tolist()[0]
        if not no_info:
            print(20 * '-')
            print('Output of model:')
            full_output = tokenizer.decode(out_ids, clean_up_tokenization_spaces=True)
            print(full_output)
            print('\nInput to the model:')
            print(history_str)
            print(20 * '-' + '\n')
        # Select part before speaker tags as answer
        i = 0
        for i, out_id in enumerate(out_ids):
            if out_id in [speaker1_tag_id, speaker2_tag_id]:
                break
        # answer = '{} {}'.format(speaker1_tag, tokenizer.decode(out_ids[:i]))
        answer = '{} {}'.format(speaker2_tag, tokenizer.decode(out_ids[:i]))
        print(answer)
        # add answer to history
        history.append(answer)
    return '\n'.join(history)


def message_sample(model, tokenizer, sample_file, speaker2_tag='<speaker2>'):
    with open(sample_file, 'r') as f:
        beginnings = [' '.join([speaker2_tag, line.strip()]) for line in f.readlines()]
    history = []
    for beginning in beginnings:
        # tokenize text and convert into vocabulary ids (input ids)
        history_enc = tokenizer.encode(beginning, add_special_tokens=True)
        with torch.no_grad():
            out_ids = sample_sequence(history_enc, model)
        out_ids = out_ids.tolist()[0]
        if tokenizer.eos_token_id in out_ids:
            eos_position = out_ids.index(tokenizer.eos_token_id)
            out_ids = out_ids[:eos_position]
        answer = tokenizer.decode(out_ids, skip_special_tokens=True)
        # add answer to history
        history.append(answer)
    return '\n'.join(history)
