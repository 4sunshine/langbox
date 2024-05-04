from math import sqrt
from tqdm import tqdm
import os
import textwrap
import fire
import json

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT_SIZE = 32
NEWS_HEAD_SIZE = (768, 256)
TEXT_SIZE = (1024, 1024)
DALLE_IMAGE_SIDE = 1024
FILTER_MAP = {
    'погибли': 'пострадали',
}


def get_font_size_for_target_size(text, target_size,
                                  font='NotoSansJP-Medium.otf', font_size=32, spacing=2.):
    if font_size is None:
        font_size = DEFAULT_FONT_SIZE
    texts = text.splitlines()
    if not texts:
        return font_size
    fnt = ImageFont.truetype(font, size=font_size, layout_engine=ImageFont.LAYOUT_RAQM)
    heights, widths = [], []
    for text in texts:
        width, height = fnt.getsize(text)
        widths.append(width)
        heights.append(height)
    mean_height = sum(heights) / len(heights)
    max_width = max(widths)
    all_heights = sum(heights) + (len(heights) - 1) * int(round(spacing * mean_height))
    h_scale = target_size[1] / all_heights
    w_scale = target_size[0] / max_width
    target_scale = min(h_scale, w_scale)
    total_height = int(round(all_heights * target_scale))
    return int(font_size * target_scale), total_height, heights[0]


def draw_side_text(target_size, text, heights, font, side_margin=0,
                   text_color='#00ff00', background_color='#00000000', anchor='ls',
                   text_back_color='#00000000', logo_map=None):
    img = Image.new('RGBA', target_size, color=background_color)
    t_draw = ImageDraw.Draw(img)

    texts = text.splitlines()

    for i, t in enumerate(texts):
        y = heights[i]
        x = side_margin

        cur_text_color = text_color

        if logo_map is not None:
            for k, v in logo_map.items():
                if k in t:
                    t = t.replace(k, '')
                    if k == '^':
                        cur_text_color = '#ffd21e'
                    elif k == '%':
                        cur_text_color = '#da2724'
                    ascent, descent = font.getmetrics()
                    (width, baseline), (offset_x, offset_y) = font.font.getsize(t)
                    target_logo_side = ascent - offset_y
                    logo = Image.open(v).convert('RGBA')
                    logo.thumbnail((target_logo_side, target_logo_side), Image.ANTIALIAS)
                    img.paste(logo, (x, int(y - target_logo_side)), mask=logo)
                    x += target_logo_side
                    break
        t_draw.text((x, y), t, anchor=anchor, fill=cur_text_color, font=font)

    return img


def draw_text_image(target_size, text,
                    font='NotoSansJP-Medium.otf',
                    text_color='#FFFFFF', background_color='#cc0000',
                    spacing=0.1, side_margin=10,
                    text_part=1.,
                    font_size=None,
                    logo_map=None,
                    return_font_size=False,
                    anchor="ls"):
    w, h = target_size
    w -= 2 * side_margin
    text_target_size = (w, h)
    font_size, total_height, h0 = get_font_size_for_target_size(text, text_target_size, font, font_size=font_size,
                                                                spacing=spacing)
    # TODO: SUPPORT FIXED FONT SIZE
    font = ImageFont.truetype(font, font_size)
    ascent, descent = font.getmetrics()
    w, h = target_size

    y_0 = int(round((h - total_height) * 0.5 + ascent))
    heights = [y_0]
    y = heights[0]
    for t in text.splitlines():
        w_t, h_t = font.getsize(t)
        y += h_t + spacing * h_t
        heights.append(y)

    if text_part < 1:
        text_len = int(round(text_part * len(text)))
        text = text[:text_len]
    img = draw_side_text(target_size, text, heights[:-1], font,
                         text_color=text_color, background_color=background_color, side_margin=side_margin,
                         logo_map=logo_map, anchor=anchor)
    if return_font_size:
        return img, font_size
    return img


def wrap_text(text, target_size, coeff=4.8, filter_map=None, append='*'):
    if filter_map is not None:
        """MANUALLY REMOVE VIOLENT CONTENT"""
        for k, v in filter_map.items():
            text = text.replace(k, v)
    total_len = len(text)
    w, h = target_size
    n_wrap = int(round(sqrt(coeff * w * total_len / h)))
    text = textwrap.wrap(text, n_wrap)
    return '\n'.join(text) + append


def prepare_news_images(prediction_file,
                        dalle_images_path,
                        news_size=(1024, 256),
                        font='Lora-Regular.ttf',
                        lower_text='* Не является достоверной информацией. Сгенерировано нейросетью.',
                        ):
    with open(prediction_file, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
    basename = os.path.splitext(os.path.basename(prediction_file))[0]
    target_dir = os.path.dirname(prediction_file)
    target_dir = os.path.join(target_dir, 'news_' + basename)
    os.makedirs(target_dir, exist_ok=True)
    images = os.listdir(dalle_images_path)
    images = sorted([os.path.join(dalle_images_path, img) for img in images if img.endswith('g')])
    imgs_per_text = len(images) // len(texts)
    for i, text in enumerate(tqdm(texts, total=len(texts))):
        cur_images = images[i * imgs_per_text: (i + 1) * imgs_per_text]
        text = wrap_text(text, news_size, filter_map=FILTER_MAP)
        text_img = draw_text_image(news_size, text, font)
        pil_images = [Image.open(img) for img in cur_images]

        for img, path in zip(pil_images, cur_images):
            w, h = img.size
            t_w, t_h = news_size
            target_w, target_h = max(w, t_w), h + t_h // 2

            if lower_text:
                lower_text_size = (t_w, t_h // 4)
                target_h += lower_text_size[1]
                lower_text_img = draw_text_image(lower_text_size, lower_text, font, background_color='#ff9e00',
                                                 text_color='#000000')

            result_img = Image.new('RGBA', (target_w, target_h))
            result_img.paste(img, (0, 0))
            result_img.paste(text_img, (0, h - t_h // 2), mask=text_img)

            if lower_text:
                result_img.paste(lower_text_img, (0, target_h - lower_text_size[1]), mask=lower_text_img)

            basename = os.path.basename(path)
            result_img.save(os.path.join(target_dir, basename))


def prepare_news_animation(prediction_file,
                           dalle_images_path,
                           news_size=(1024, 256),
                           font='Lora-Regular.ttf',
                           lower_text='* Не является достоверной информацией. Сгенерировано нейросетью.',
                           fps=60,
                           news_duration=5,
                           text_birth_duration=3,
                           transition_duration=0.15,
                           background_image=None,
                           overlay_image=None,
                           news_head=None,
                           news_head_font=None
                           ):

    total_frames = int(fps * news_duration)
    text_birth_frames = int(fps * text_birth_duration)
    transition_frames = int(fps * transition_duration)

    with open(prediction_file, 'r') as f:
        texts = [line.strip() for line in f.readlines()]
    basename = os.path.splitext(os.path.basename(prediction_file))[0]
    target_dir = os.path.dirname(prediction_file)
    target_dir = os.path.join(target_dir, 'animation_' + basename)
    os.makedirs(target_dir, exist_ok=True)
    images = os.listdir(dalle_images_path)
    images = sorted([os.path.join(dalle_images_path, img) for img in images if img.endswith('g')])
    imgs_per_text = len(images) // len(texts)

    if background_image is not None:
        background_image = Image.open(background_image)
        bw, bh = background_image.size

        if overlay_image is not None:
            overlay_image = Image.open(overlay_image)

        if news_head is not None:
            with open(news_head, 'r') as f:
                news_head_text = f.read()

            start_position = (int((bw - DALLE_IMAGE_SIDE) / 2), int((bh - NEWS_HEAD_SIZE[1]) / 3))

            for i in range(fps):
                news_head_img = draw_text_image(NEWS_HEAD_SIZE, news_head_text, news_head_font,
                                                text_part=min(2 * i / fps, 1))

                k_coeff = 20
                delta = max(2 * i / fps - 1, 0)
                paste_position = (int(start_position[0] + k_coeff * delta),
                                  int(start_position[1] + k_coeff * 1.5 * delta))

                news_head_final = background_image.copy()
                news_head_final.paste(news_head_img, paste_position)

                if delta > 0:
                    r, g, b, a = overlay_image.split()
                    a = a.point(lambda x: x * delta)
                    overlay_image_copy = Image.merge('RGBA', [r, g, b, a])
                    news_head_final = Image.alpha_composite(news_head_final, overlay_image_copy)

                news_head_final.save(os.path.join(target_dir, f'head_{i:04d}.png'))

    for i, text in enumerate(tqdm(texts, total=len(texts))):
        cur_images = images[i * imgs_per_text: (i + 1) * imgs_per_text]
        text = wrap_text(text, news_size, filter_map=FILTER_MAP)
        pil_images = [Image.open(img) for img in cur_images]

        prev_blend = None
        next_blend = pil_images[1] if len(pil_images) > 1 else None

        frames_per_image = int(total_frames / len(pil_images))

        prev_img_index = 0
        for t in tqdm(range(total_frames)):
            image_index = int(t / frames_per_image)
            img = pil_images[image_index]

            if prev_img_index != image_index:
                prev_img_index = image_index
                prev_blend = pil_images[max(image_index - 1, 0)]
                next_blend = pil_images[min(image_index + 1, len(pil_images) - 1)]

            transition_start_fraction = (t % frames_per_image) / transition_frames
            transition_end_fraction = (((image_index + 1) * frames_per_image - t) % frames_per_image) / transition_frames
            if transition_start_fraction < 1:
                blend_img = prev_blend
                alpha = 0.5 * (1 - transition_start_fraction)
            elif transition_end_fraction < 1:
                blend_img = next_blend
                alpha = 0.5 * (1 - transition_end_fraction)
            else:
                blend_img = None
                alpha = 0

            if blend_img is not None:
                final_img = Image.blend(img, blend_img, alpha=alpha)
            else:
                final_img = img

            text_part = t / text_birth_frames
            text_img = draw_text_image(news_size, text, font, text_part=text_part)

            w, h = img.size
            t_w, t_h = news_size
            target_w, target_h = max(w, t_w), h + t_h // 2

            if lower_text:
                lower_text_size = (t_w, t_h // 4)
                target_h += lower_text_size[1]
                lower_text_img = draw_text_image(lower_text_size, lower_text, font, background_color='#ff9e00',
                                                 text_color='#000000')

            result_img = Image.new('RGBA', (target_w, target_h))
            result_img.paste(final_img, (0, 0))
            result_img.paste(text_img, (0, h - t_h // 2), mask=text_img)

            if lower_text:
                result_img.paste(lower_text_img, (0, target_h - lower_text_size[1]), mask=lower_text_img)

            if background_image is not None:
                final = background_image.copy()
                final.paste(result_img, (int((bw - target_w) / 2), int((bh - target_h) / 2)))

                if overlay_image is not None:
                    if (transition_start_fraction < 1) and (image_index == 0):
                        r, g, b, a = overlay_image.split()
                        a = a.point(lambda x: x * (1 - transition_start_fraction))
                        overlay_image_copy = Image.merge('RGBA', [r, g, b, a])
                        final = Image.alpha_composite(final, overlay_image_copy)
                    elif (transition_end_fraction < 1) and (image_index == (len(pil_images) - 1))\
                            and (transition_start_fraction > 1):
                        r, g, b, a = overlay_image.split()
                        a = a.point(lambda x: x * (1. - transition_end_fraction))
                        overlay_image_copy = Image.merge('RGBA', [r, g, b, a])
                        final = Image.alpha_composite(final, overlay_image_copy)
                result_img = final

            basename = f'item_{i:04d}_{t:05d}.png'
            result_img.save(os.path.join(target_dir, basename))


def animate_text(text_file,
                 font='NotoSansJP-Medium.otf',
                 background_color='#000000',
                 text_color='#FFFFFF',
                 target_size=(1080, 1920),
                 fps=60,
                 duration=2,
                 birth=1,
                 wrap=False,
                 logo_map=None):
    total_frames = fps * duration
    birth_frames = fps * birth

    with open(text_file, 'r') as f:
        text = f.read()
    if wrap:
        text = wrap_text(text, TEXT_SIZE, append='')

    if logo_map is not None:
        with open(logo_map, 'r') as f:
            logo_map = json.load(f)

    basename = os.path.splitext(os.path.basename(text_file))[0]
    target_dir = os.path.dirname(text_file)
    target_dir = os.path.join(target_dir, 'animation_' + basename)
    os.makedirs(target_dir, exist_ok=True)

    background_image = Image.new('RGBA', target_size, background_color)
    bw, bh = background_image.size

    for i in range(total_frames):
        text_img = draw_text_image(TEXT_SIZE, text, font, text_color=text_color, background_color=background_color,
                                   text_part=min(i / birth_frames, 1), logo_map=logo_map)

        paste_position = (int((bw - TEXT_SIZE[0]) / 2), int((bh - TEXT_SIZE[1]) / 2))

        news_head_final = background_image.copy()
        news_head_final.paste(text_img, paste_position)

        news_head_final.save(os.path.join(target_dir, f'text_{i:04d}.png'))


def convert_rgba_to_rgb(folder):
    all_files = os.listdir(folder)
    all_files = [f for f in all_files if f.endswith('.png')]
    target_dir = os.path.join(os.path.dirname(folder), 'rgb_' + os.path.basename(folder))
    os.makedirs(target_dir, exist_ok=True)
    for f in tqdm(all_files, total=len(all_files)):
        img = Image.open(os.path.join(folder, f)).convert('RGB')
        img.save(os.path.join(target_dir, f))


if __name__ == '__main__':
    fire.Fire(prepare_news_images)
    # fire.Fire(convert_rgba_to_rgb)
    # fire.Fire(animate_text)
    # fire.Fire(prepare_news_animation)
