import torch
from tqdm import tqdm, trange
import argparse
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    get_linear_schedule_with_warmup, AutoModelWithLMHead, AutoTokenizer
)

from dataset.utils import set_seed, add_special_tokens_
from dataset.text_dataset import get_data_loader
from utils.sample import sample_sequence
import logging
import tensorboardX
import os

# set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description='Training params')
    parser.add_argument('--run_name', default='telegram', type=str)
    parser.add_argument('--input_file', default='chat.txt', type=str)
    parser.add_argument('--sampling_history', default=None, type=str)
    parser.add_argument('--val_file', default=None, type=str)
    parser.add_argument('--model_type', default='gpt2', type=str)
    parser.add_argument('--save_every', default=30, type=int)
    parser.add_argument('--max_input_lenth', default=400, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--train_batch_size', default=2, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int)
    parser.add_argument('--warmup_steps', default=8, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    parser.add_argument('--max_norm', default=1, type=float)
    parser.add_argument('--n_epochs', default=12, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--always_save', default=1, type=int)
    return parser.parse_args()


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
    # history = [f"{speaker2_tag} Война на Украине.\n", f"{speaker2_tag} Спецоперация войск РФ.\n",
    #            f"{speaker2_tag} Путин дал жёсткий ответ западным санкциям.\n",
    #            f"{speaker2_tag} Зеленский выступил с обращением к нации.\n"]
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


def validate(model, data_loader, global_step, writer, device='cuda'):
    model.eval()
    av_loss = 0
    with torch.no_grad():
        pbar = tqdm(data_loader, position=0, desc='evaluation')  # progress bar
        for step, batch in enumerate(pbar):
            # Skip past any already trained steps if resuming training
            # the language model targets (labels) are the same as the input!
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, labels=labels)
            loss = output['loss']
            val_loss = loss.item()
            # Compute a running average of the loss
            av_loss = (step * av_loss + val_loss) / (step + 1)
    writer.add_scalar('val_loss', av_loss, global_step=global_step)
    return av_loss


def main(cfg):
    set_seed(cfg.seed)
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    train_batch_size = cfg.train_batch_size
    model_type = cfg.model_type
    n_epochs = cfg.n_epochs
    device = cfg.device
    max_norm = cfg.max_norm
    save_every = cfg.save_every

    base_output_dir = os.path.join('runs', cfg.run_name)
    os.makedirs(base_output_dir, exist_ok=True)
    log_dir = os.path.join(base_output_dir, 'tensorboard')
    writer = tensorboardX.SummaryWriter(logdir=log_dir)

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer  #if "gpt2" in model_type else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_type)
    # Load model
    model_class = AutoModelWithLMHead  #GPT2LMHeadModel  #if "gpt2" in model_type else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(model_type)
    model.to(device)

    add_special_tokens_(model, tokenizer)

    # Get data loaders
    logger.info("Prepare datasets")
    data_loader = get_data_loader(tokenizer, cfg.input_file, train_batch_size=train_batch_size)
    if cfg.val_file is not None:
        val_data_loader = get_data_loader(tokenizer, cfg.val_file, shuffle=False, train_batch_size=train_batch_size)
    else:
        val_data_loader = None
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr, eps=cfg.adam_epsilon)
    t_total = len(data_loader) // gradient_accumulation_steps * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if we are training from a checkpoint or from a pretrained model
    if os.path.exists(model_type):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(model_type.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(data_loader) // gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(data_loader) // gradient_accumulation_steps)
        logger.info("Continuing training from checkpoint, will skip to saved global_step")
        logger.info(f"Continuing training from epoch {epochs_trained}")
        logger.info(f"Continuing training from global step {global_step}")
        logger.info(f"Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")

    # Training loop
    model.zero_grad()
    epoch_pbar = trange(epochs_trained, int(n_epochs))  # epoch progress bar
    av_loss = 0
    best_val_loss = 1e10
    best_epoch = 0
    for current_epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch [{current_epoch + 1}/{n_epochs}]")  # description of epoch progress bar
        pbar = tqdm(data_loader, position=0)  # progress bar
        for step, batch in enumerate(pbar):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            # the language model targets (labels) are the same as the input!
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs, labels=labels)
            loss, logits, past_key_values = output['loss'], output['logits'], output['past_key_values']
            loss.backward()
            tr_loss = loss.item()
            # Compute a running average of the loss
            av_loss = (step * av_loss + tr_loss) / (step + 1)
            writer.add_scalar('train_loss', av_loss, global_step=global_step)
            pbar.set_description(f"Average loss: {av_loss:.4f}")
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % save_every == 0 and global_step > 0:
                    if cfg.always_save:
                        checkpoint_prefix = "checkpoint"
                        output_dir = os.path.join(base_output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        logger.info(f"Saving model checkpoint to {output_dir}")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f"Saving optimizer and scheduler states to {output_dir}")
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    model.eval()
                    print('*** START SAMPLING ***')
                    history = sample(model, tokenizer, cfg.sampling_history)
                    writer.add_text('Sample', history, global_step=global_step)
                    print('*** FINISH SAMPLING ***')

                    if val_data_loader is not None:
                        val_loss = validate(model, val_data_loader, global_step, writer, device)
                        logger.info(f"Current val loss: {val_loss}")
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = current_epoch
                            output_dir = os.path.join(base_output_dir, 'best')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            logger.info(f"Saving best checkpoint to {output_dir}")
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                        logger.info(f"Current epoch: {current_epoch}. Best epoch: {best_epoch}.")

    # save model
    output_dir = os.path.join(base_output_dir, 'final')
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    config = parse_args()
    main(config)


