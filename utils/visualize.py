from math import sqrt
from tqdm import tqdm
import os
import textwrap
import fire

from PIL import Image, ImageDraw, ImageFont


def get_font_size_for_target_size(text, target_size,
                                  font='NotoSansJP-Medium.otf', font_size=32, spacing=2):
    texts = text.splitlines()
    if not texts:
        return 32
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
                   text_back_color='#00000000'):
    img = Image.new('RGBA', target_size, color=background_color)
    t_draw = ImageDraw.Draw(img)

    texts = text.splitlines()

    for i, t in enumerate(texts):
        y = heights[i]
        x = side_margin
        t_draw.text((x, y), t, anchor=anchor, fill=text_color, font=font)

    return img


def draw_text_image(target_size, text,
                    font='NotoSansJP-Medium.otf',
                    text_color='#FFFFFF', background_color='#cc0000',
                    text_back_color='#00000000', spacing=0.1,
                    font_size=None):

    if font_size is None:
        font_size, total_height, h0 = get_font_size_for_target_size(text, target_size, font, font_size=32,
                                                                    spacing=spacing)
    # TODO: SUPPORT FIXED FONT SIZE
    font = ImageFont.truetype(font, font_size)
    ascent, descent = font.getmetrics()
    w, h = target_size

    # t_draw.line((0, h / 2, w, h / 2), fill='#abb2b9')
    y_0 = int(round((h - total_height) * 0.5 + ascent))
    heights = [y_0]
    y = heights[0]
    for t in text.splitlines():
        w_t, h_t = font.getsize(t)
        # (width, baseline), (offset_x, offset_y) = font.font.getsize(t)
        y += h_t + spacing * h_t
        heights.append(y)

    # for i in range(1, len(text) + 1):
    #     cur_text = text[:i]
    #     print(cur_text)
    img = draw_side_text(target_size, text, heights[:-1], font,
                         text_color=text_color, background_color=background_color)
    return img


def wrap_text(text, target_size, coeff=4.8):
    total_len = len(text)
    w, h = target_size
    n_wrap = int(round(sqrt(coeff * w * total_len / h)))
    text = textwrap.wrap(text, n_wrap)
    return '\n'.join(text)


def prepare_news_images(prediction_file,
                        dalle_images_path,
                        news_size=(1024, 256),
                        font='Lora-Regular.ttf'):
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
        text = wrap_text(text, news_size)
        text_img = draw_text_image(news_size, text, font)
        pil_images = [Image.open(img) for img in cur_images]
        for img, path in zip(pil_images, cur_images):
            w, h = img.size
            t_w, t_h = news_size
            target_w, target_h = max(w, t_w), h + t_h
            result_img = Image.new('RGBA', (target_w, target_h))
            result_img.paste(img, (0, 0))
            result_img.paste(text_img, (0, h), mask=text_img)
            basename = os.path.basename(path)
            result_img.save(os.path.join(target_dir, basename))


if __name__ == '__main__':
    fire.Fire(prepare_news_images)
