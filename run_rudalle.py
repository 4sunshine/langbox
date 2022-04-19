"""CODE BASED ON: https://github.com/ai-forever/ru-dalle"""

import multiprocessing
import os

import torch
from psutil import virtual_memory
import transformers
import more_itertools
from tqdm.auto import tqdm

from rudalle.pipelines import show, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_ruclip
from rudalle.utils import seed_everything, torch_tensors_to_pil_list


def memory_check(allowed_memory=7.5):
    total_memory = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total_memory < allowed_memory:
        raise MemoryError
    print('Total GPU RAM:', round(total_memory, 2), 'Gb')

    ram_gb = round(virtual_memory().total / 1024**3, 1)
    print('CPU:', multiprocessing.cpu_count())
    print('RAM GB:', ram_gb)
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    if torch.__version__ >= '1.8.0':
        k = allowed_memory / total_memory
        torch.cuda.set_per_process_memory_fraction(k, 0)
        print('Allowed GPU RAM:', round(allowed_memory, 2), 'Gb')
        print('GPU part', round(k, 4))


def get_models():
    # prepare models:
    device = 'cuda'
    dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
    tokenizer = get_tokenizer()
    vae = get_vae(dwt=True)
    # prepare utils:
    ruclip, ruclip_processor = get_ruclip('ruclip-vit-base-patch32-v5')
    return dalle, tokenizer, vae, ruclip, ruclip_processor

def generate_codebooks(text, tokenizer, dalle, top_k, top_p, images_num, image_prompts=None, temperature=1.0, bs=8,
                    seed=None, use_cache=True):
    vocab_size = dalle.get_param('vocab_size')
    text_seq_length = dalle.get_param('text_seq_length')
    image_seq_length = dalle.get_param('image_seq_length')
    total_seq_length = dalle.get_param('total_seq_length')
    device = dalle.get_param('device')
    text = text.lower().strip()
    input_ids = tokenizer.encode_text(text, text_seq_length=text_seq_length)
    codebooks = []
    for chunk in more_itertools.chunked(range(images_num), bs):
        chunk_bs = len(chunk)
        with torch.no_grad():
            attention_mask = torch.tril(torch.ones((chunk_bs, 1, total_seq_length, total_seq_length), device=device))
            out = input_ids.unsqueeze(0).repeat(chunk_bs, 1).to(device)
            has_cache = False
            if image_prompts is not None:
                prompts_idx, prompts = image_prompts.image_prompts_idx, image_prompts.image_prompts
                prompts = prompts.repeat(chunk_bs, 1)
            for idx in tqdm(range(out.shape[1], total_seq_length)):
                idx -= text_seq_length
                if image_prompts is not None and idx in prompts_idx:
                    out = torch.cat((out, prompts[:, idx].unsqueeze(1)), dim=-1)
                else:
                    logits, has_cache = dalle(out, attention_mask,
                                              has_cache=has_cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, vocab_size:]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)
            codebooks.append(out[:, -image_seq_length:].cpu())
    return codebooks


def prepare_codebooks(text, tokenizer, dalle, dalle_bs=4, seed=6995):
    seed_everything(seed)
    codebooks = []
    for top_k, top_p, images_num in [
        (2048, 0.995, 8),
        (1536, 0.99, 8),
        (1024, 0.99, 8),
        (1024, 0.98, 8),
        (512, 0.97, 8),
        (384, 0.96, 8),
        (256, 0.95, 8),
        (128, 0.95, 8),
    ]:
        codebooks += generate_codebooks(text, tokenizer, dalle, top_k=top_k, images_num=images_num, top_p=top_p,
                                        bs=dalle_bs)
    return codebooks

def synth_images(codebooks):
    pil_images = []
    for _codebooks in tqdm(torch.cat(codebooks).cpu()):
        with torch.no_grad():
            images = vae.decode(_codebooks.unsqueeze(0))
            pil_images += torch_tensors_to_pil_list(images)
    return pil_images

def get_sr_model(dalle, device='cuda'):
    from rudalle import get_realesrgan

    dalle = dalle.to('cpu')
    del dalle
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    realesrgan = get_realesrgan('x2', device=device, fp16=True)
    return realesrgan

def upsample_sr(pil_images, realesrgan):
    from rudalle.pipelines import super_resolution
    sr_images = super_resolution(pil_images, realesrgan, batch_size=1)
    return sr_images


def create_top_n_images(text, topk=6):
    memory_check()
    models = get_models()
    dalle, tokenizer, vae, ruclip, ruclip_processor = models
    codebooks = prepare_codebooks(text, tokenizer, dalle)
    pil_images = synth_images(codebooks)
    top_images, clip_scores = cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device='cpu', count=topk)
    realesrgan = get_sr_model(dalle)
    sr_images = upsample_sr(top_images, realesrgan)
    return sr_images


def generate_by_texts(text_file):
    with open(text_file, 'r') as f:
        texts = [line.strip() for line in f.readlines()]

    dirname = os.path.dirname(text_file)
    basename = os.path.basename(text_file)
    basename = os.path.splitext(basename)[0]

    target_dir = os.path.join(dirname, 'dalle_' + basename)
    os.makedirs(target_dir, exist_ok=True)

    for i, text in enumerate(texts):
        print(f'Dalle generation for text {i + 1} out of {len(texts)}')
        pil_images = create_top_n_images(text)
        text_basename = f'text_{i:04d}'
        with open(os.path.join(target_dir, text_basename + '.txt'), 'w') as f:
            f.write(text)
        for j, img in enumerate(pil_images):
            image_basename = text_basename + f'_{j:04d}.png'
            img.save(os.path.join(target_dir, image_basename))


if __name__ == '__main__':
    output_dir = 'images_woman'
    os.makedirs(output_dir, exist_ok=True)
    memory_check()
    text = 'Фотография идеальная женщина'
    models = get_models()
    dalle, tokenizer, vae, ruclip, ruclip_processor = models
    codebooks = prepare_codebooks(text, tokenizer, dalle)
    pil_images = synth_images(codebooks)
    top_images, clip_scores = cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device='cpu', count=6)
    realesrgan = get_sr_model(dalle)
    sr_images = upsample_sr(top_images, realesrgan)
    for i, img in enumerate(sr_images):
        img.save(os.path.join(output_dir, f'img_sr_{i:02d}.png'))
