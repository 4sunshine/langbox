import json
import pickle
from datetime import timedelta, datetime
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageSequence
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import centroid
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import imageio
import wikipedia as wp
import pandas as pd
from dateutil import parser
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from copy import deepcopy
from descartes import PolygonPatch
from pydub import AudioSegment
from pydub.playback import play
import random
from functools import partial
import subprocess
from collections import deque
from difflib import get_close_matches
from glob import glob
import cv2
import seaborn as sns

FRAMES_PER_MONTH = 3
FPS = 30
EVIL_AT_TARGET_MONTHS = 1
EVIL_AT_TARGET = EVIL_AT_TARGET_MONTHS * FRAMES_PER_MONTH
SECONDS_PER_MONTH = round(FRAMES_PER_MONTH / FPS, 1)
SEED = 6743
random.seed(SEED)
VERTICAL_MARGIN = 40
LEFT_OFFSET = 1
BACKGROUND_COLOR = "#aaffbb"
WORLD_COLOR = "#006633"

WORLD_COLOR_0 = "#C71585"
BACKGROUND_COLOR_0 = "#FFB6C1"

DURATIONS_TO_FORGET = 3
MIN_MONTHS_FORGET = 36
MAX_MONTHS_FORGET = 60

AUDIO_CROSSFADE_DURAION = 1000  # ms

EVIL_COLOR = '#0a0909' #'#21618c'#'navy'
SAD_COLOR = '#ff2f2f'
EDGE_COLOR = '#ff2f2f'
ALPHA_COUNTRY = 0.9

COMBINED_AUDIO_NAME = f"combined_{SEED}.mp3"
MOVIE_NAME = f"movie_{SEED}.mp4"
OUTPUT_NAME = f"output_{SEED}.mp4"
REST_AUDIO_NAME = f"rest_music_{SEED}.mp3"
MOVIE_SOUND = f"movie_sound_{SEED}.mp3"

ROUND_TO_MONTHS = True


NOW = datetime(year=2024, month=4, day=1)  #datetime.now()


def calculate_ms(value_months):
    return int(round(1000 * value_months * SECONDS_PER_MONTH))


def frames_to_ms(i):
    return int(round(1000 * i / FPS))


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


replacements = {
    "United States": "USA",
    "Iraq": "IRQ",
    "Kuwait": "KWT",
    "Croatia": "HRV",
    "Yugoslavia": "SRB",
    "Yugoslavia[g]": "SRB",
    "Slovenia": "SVN",
    "Bosnia and Herzegovina": "BIH",
    "Haiti": "HTI",
    "Peru": "PER",
    "Ecuador": "ECU",
    "Ethiopia": "ETH",
    "Eritrea": "ERI",
    "Kosova": "KOS",
    "Russia": "RUS",
    "India": "IND",
    "Pakistan": "PAK",
    "Ichkeria": "RUS",
    "Zaire": "COG",
    "AFDL": "COG",
    "Rwanda": "RWA",
    "Uganda": "UGA",
    "Islamic Emirate of Afghanistan": "AFG",
    "Afghanistan": "AFG",
    "Israel": "ISR",
    "Gaza Strip": "PSX",
    "Georgia": "GEO",
    "Sudan": "SDN",
    "South Sudan": "SDS",
    "Libya": "LBY",
    "NATO": "USA",
    "Ukraine": "UKR",
    "Djibouti": "DJI",
    "Iran": "SYR",  ### Actually actions take place in Syria
    "Syria": "SYR",
    "Azerbaijan": "AZE",
    "Artsakh": "ARM",
    "Turkey": "TUR",
    "North and East Syria": "SYR",
    "Armenia": "ARM",
    "Kyrgyzstan": "KGZ",
    "Tajikistan": "TJK",
    "Qatar": "QAT",
    "Sweden": "SWE",
    "United Arab Emirates": "ARE",
    "Australia": "AUS",
    "United Kingdom": "GBR",
    "Canada": "CAN",
    "Poland": "POL",
    "France[f]": "FRA",
    "Burundi": "BDI",
    "Angola SPLA": "AGO",
}

## TODO: actually needs to be improved
LIST_NATO = {
    1990: ["USA", "BEL", "ITA", "LUX", "ESP", "NLD", "DEU", "GRC",
           "ISL", "TUR", "DNK", "NOR", "GBR", "PR1", "CAN", "FRA"],
    1999: ["POL", "CZE", "HUN"],
    2004: ["LVA", "SVN", "EST", "LTU", "ROU", "BGR", "SVK"],
    2009: ["HRV"],
    2017: ["MNE"],
    2020: ["MKD"],
    2023: ["FIN"],
    2024: ["SWE"],
}


def get_conflict_data(start, finish, side_a, side_b, index, is_nato_a=False):
    start = parser.parse(start)
    finish = parser.parse(finish)

    global ROUND_TO_MONTHS

    if ROUND_TO_MONTHS:
        start = start.replace(day=1)
        finish = finish.replace(day=1)

    duration = max(diff_month(finish, start), 1)
    forget_after_months = min(MAX_MONTHS_FORGET, max(MIN_MONTHS_FORGET, DURATIONS_TO_FORGET * duration))
    forget_after = finish + relativedelta(months=forget_after_months)
    total_duration = forget_after_months + duration

    data = {
        "side_a": side_a,
        "side_b": side_b,
        "start": start,
        "finish": finish,
        "num_months": duration,
        "index": index,
        "forget_after": forget_after,
        "total_duration": total_duration,
        "is_nato_a": is_nato_a,
    }

    return data


def get_conflicts_from_table(wikipage):
    try:
        df = pd.read_html(wikipage)[1]  # Try 2nd table first as most pages contain contents table first
    except IndexError:
        df = pd.read_html(wikipage)[0]

    df.reset_index()
    conflicts = []
    for index, row in df.iterrows():
        states = row["States in conflict"].to_string()
        offense, defence = states.split("\n")[:2]
        offense = offense.strip().replace("States in conflict", "").strip()
        defence = defence.strip().replace("States in conflict.1", "").strip()
        start = row["Start"].to_string().split("\n")[:1][0]
        start = start.strip().replace("Start", "").strip()
        finish = row["Finish"].to_string().split("\n")[:1][0]
        finish = finish.strip().replace("Finish", "").strip()

        is_nato_a = False

        if ("nato" in offense.lower()) or ("kosova" in offense.lower()):
            is_nato_a = True

        offense = offense.split("\xa0")
        offense = [of.strip() for of in offense if "..." not in of]
        defence = defence.split("\xa0")
        defence = [of.strip() for of in defence if "..." not in of]
        #
        if finish.lower() == "ongoing":
            finish = "1 January 2025"

        side_a = offense[0]

        side_b = defence[0]

        for rep in replacements:
            if side_a.strip() == rep:
                side_a = replacements[rep]
            if side_b.strip() == rep:
                side_b = replacements[rep]

        if (side_a == "UKR") and (side_b == "RUS"):
            # dividing 3 stages of conflict
            start_1 = start
            finish_1 = "23 February 2022"
            start_2 = "24 February 2022"
            finish_2 = finish
            side_a_1 = side_a
            side_b_1 = side_a
            side_a_2 = side_b
            side_b_2 = side_a
            start_3 = "30 September 2022"
            finish_3 = finish
            side_a_3 = side_a
            side_b_3 = side_b
            data_1 = get_conflict_data(start_1, finish_1, side_a_1, side_b_1, index)
            data_2 = get_conflict_data(start_2, finish_2, side_a_2, side_b_2, index)
            data_3 = get_conflict_data(start_3, finish_3, side_a_3, side_b_3, index)
            conflicts.append(data_1)
            conflicts.append(data_2)
            conflicts.append(data_3)
            continue

        if (len(offense) == 1) or ("KOS" == side_a):

            data = get_conflict_data(start, finish, side_a, side_b, index, is_nato_a=is_nato_a)

            conflicts.append(data)

        else:
            result_side_a = [side_a]

            for offn in offense[1:]:
                for rep in replacements:
                    if offn.strip() == rep:
                        offn = replacements[rep]
                        result_side_a.append(offn)
                        break
            if len(result_side_a) == 1:
                result_side_a = side_a
            data = get_conflict_data(start, finish, result_side_a, side_b, index, is_nato_a=is_nato_a)
            conflicts.append(data)

    for conf in conflicts:
        print("$$$$$$$")
        print(conf)

    return conflicts


def make_audio_for_conflict(conflict, evil_segment, cry_segment, shoot_is, shoot_modern, shoot_urban):
    total_duration = conflict["total_duration"]
    num_months = conflict["num_months"]

    num_months_ms = calculate_ms(num_months)
    total_duration_ms = calculate_ms(total_duration)
    evil_at_target_ms = calculate_ms(EVIL_AT_TARGET_MONTHS)
    base = AudioSegment.silent(duration=total_duration_ms)
    overlay = base.overlay(evil_segment)
    if conflict["side_b"] in ("AFG", "IRQ", "SYR", "PSX"):
        overlay_shoot = shoot_is
    else:
        overlay_shoot = shoot_modern if random.random() < 0.5 else shoot_urban

    # overlay_shoot -= 5  # reduce shoot volume

    start_pos = random.randint(0, len(overlay_shoot) - num_months_ms - 1)
    overlay = overlay.overlay(overlay_shoot[start_pos: start_pos + num_months_ms - evil_at_target_ms],
                              position=evil_at_target_ms)
    overlay = overlay.overlay(cry_segment[:total_duration_ms - evil_at_target_ms], position=evil_at_target_ms)

    return overlay


def get_conflict_by_date(date, conflicts):
    return [conf for conf in conflicts if (conf["start"] <= date) and (conf["forget_after"] >= date)]


def get_country_point(su_a3, world):
    country = world[world["SU_A3"] == su_a3]

    geometry = country["geometry"]
    c_centroid = centroid(geometry)
    coords = (c_centroid.x.item(), c_centroid.y.item())
    return coords, c_centroid, geometry


def set_image(image, i):
    j = 0
    iterator = ImageSequence.Iterator(image)
    while True:
        try:
            frame = next(iterator)
        except:
            iterator = ImageSequence.Iterator(image)
            frame = next(iterator)

        if j == i:
            break

        j += 1

    return frame.convert("RGBA")


def parse_ua_db(ua_regions_db):
    ua = gpd.read_file(ua_regions_db)

    # crimea_geom = ukr[ukr["ADM1_PCODE"] == "UA01"]["geometry"]
    zpr_geom = ua[ua["ADM1_PCODE"] == "UA23"]["geometry"]
    lnr_geom = ua[ua["ADM1_PCODE"] == "UA44"]["geometry"]
    khe_geom = ua[ua["ADM1_PCODE"] == "UA65"]["geometry"]
    dnr_geom = ua[ua["ADM1_PCODE"] == "UA14"]["geometry"]

    return zpr_geom, lnr_geom, khe_geom, dnr_geom


def world_sep_2022(world, ua_dbf):
    zpr_geom, lnr_geom, khe_geom, dnr_geom = parse_ua_db(ua_dbf)
    rus_geom = world[world["SU_A3"] == "RUS"]["geometry"]

    ukr_geom = world[world["SU_A3"] == "UKR"]["geometry"]
    merged_polys = unary_union([rus_geom, zpr_geom, lnr_geom, khe_geom, dnr_geom])
    world.loc[world["SU_A3"] == "RUS", ["geometry"]] = merged_polys

    rus_geom = world[world["SU_A3"] == "RUS"]["geometry"]
    ukr_series = gpd.GeoSeries(ukr_geom.iloc[0])
    rus_series = gpd.GeoSeries(rus_geom.iloc[0])

    mod_ua = ukr_series.difference(rus_series)

    upd_ua_b = MultiPolygon([p for p in mod_ua.item().geoms if p.area > 1])

    add_to_russia = MultiPolygon([p for p in mod_ua.item().geoms if p.area <= 1])
    rus_geom = world[world["SU_A3"] == "RUS"]["geometry"]
    merged_polys_russia = unary_union([add_to_russia, rus_geom.item()])

    world.loc[world["SU_A3"] == "UKR", ["geometry"]] = upd_ua_b
    world.loc[world["SU_A3"] == "RUS", ["geometry"]] = merged_polys_russia
    return world


def add_img_to_plot(im, i, coords, zoom=0.1):
    img_evil = set_image(im, i)
    img_evil = OffsetImage(img_evil, zoom=zoom)
    return AnnotationBbox(img_evil, coords, frameon=False)


def main(folder):
    images = {
        "evil": "evil.png",
        "cry": "cry.gif",
        "heart": "heart.png",
        "shoot": "shoot.png"
    }

    sounds = {
        "evil": "evil.mp3",
        "cry": "cry.mp3",
        "shoot_is": "shoot_is.mp3",
        "shoot_urban": "shoot_urban.mp3",
        "shoot_modern": "shoot_modern.mp3"
    }

    epochs = tuple(yy for yy in range(1990, NOW.year + 1, 10))

    evil_segment = AudioSegment.from_mp3(os.path.join(folder, sounds["evil"]))
    cry_segment = AudioSegment.from_mp3(os.path.join(folder, sounds["cry"]))
    shoot_is = AudioSegment.from_mp3(os.path.join(folder, sounds["shoot_is"]))
    shoot_urban = AudioSegment.from_mp3(os.path.join(folder, sounds["shoot_urban"]))
    shoot_modern = AudioSegment.from_mp3(os.path.join(folder, sounds["shoot_modern"]))

    epoch_sounds = [AudioSegment.from_mp3(os.path.join(folder, str(epoch), "output.mp3")) for epoch in epochs]

    im_evil = Image.open(os.path.join(folder, images["evil"]))
    im_cry = Image.open(os.path.join(folder, images["cry"]))
    im_heart = Image.open(os.path.join(folder, images["heart"]))
    im_shoot = Image.open(os.path.join(folder, images["shoot"]))

    world_dbf = os.path.join(folder, "ne_110m_admin_0_countries/ne_110m_admin_0_countries.dbf")
    world = gpd.read_file(world_dbf)

    ua_dbf = os.path.join(folder, "ukr_admbnda_sspe_20230201_SHP/ukr_admbnda_adm1_sspe_20230201.dbf")
    world_sep22 = world_sep_2022(deepcopy(world), ua_dbf)

    fig, ax = plt.subplots(dpi=300, frameon=False)

    html = os.path.join(folder, "List of interstate wars since 1945 - Wikipedia.html")
    conflicts = get_conflicts_from_table(html)

    prepare_audio_fn = partial(make_audio_for_conflict, evil_segment=evil_segment, cry_segment=cry_segment,
                               shoot_is=shoot_is, shoot_modern=shoot_modern, shoot_urban=shoot_urban)

    conflicts = [{**conflict, "audio": prepare_audio_fn(conflict)} for conflict in conflicts]

    def animate_conflict(i, num_months, side_a, side_b, group_a=None):

        if isinstance(side_a, list):
            if group_a is not None:
                group_a += [csa for csa in side_a if csa not in group_a]
            else:
                group_a = side_a

            side_a = side_a[0]

        def plotCountryPatch(axes, country_name, fcolor):
            # plot a country on the provided axes
            nami = world[world["SU_A3"] == country_name]
            namigm = nami.__geo_interface__['features']  # geopandas's geo_interface
            namig0 = {'type': namigm[0]['geometry']['type'], \
                      'coordinates': namigm[0]['geometry']['coordinates']}
            axes.add_patch(PolygonPatch(namig0, fc=fcolor, alpha=ALPHA_COUNTRY,
                                        edgecolor=EDGE_COLOR, zorder=2,
                                        linestyle=''))

        plotCountryPatch(ax, side_b, SAD_COLOR)

        point_coords_source, point, geom = get_country_point(side_a, world)

        point_coords_target, point_target, geom_target = get_country_point(side_b, world)

        trajectory_evil = np.linspace(np.array(point_coords_source),
                                      np.array(point_coords_target),
                                      num=EVIL_AT_TARGET)

        max_position_evil = trajectory_evil[min(int(round(0.9 * EVIL_AT_TARGET)), EVIL_AT_TARGET - 1)]

        if i >= EVIL_AT_TARGET:
            point_coords_evil = max_position_evil
        else:
            point_coords_evil = trajectory_evil[i]

        is_shooting = i <= num_months * FRAMES_PER_MONTH

        if is_shooting:
            if group_a is None:
                plotCountryPatch(ax, side_a, EVIL_COLOR)
            else:
                for side in group_a:
                    plotCountryPatch(ax, side, EVIL_COLOR)
            ab_shoot = add_img_to_plot(im_shoot, i, point_coords_evil, zoom=0.2)
            ax.add_artist(ab_shoot)

        ab_evil = add_img_to_plot(im_evil, i, point_coords_source)
        ax.add_artist(ab_evil)

        is_cry = i >= EVIL_AT_TARGET
        if is_cry:
            ab_cry = add_img_to_plot(im_cry, i - EVIL_AT_TARGET, point_coords_target)
            ax.add_artist(ab_cry)

        return {
            "cry": is_cry,
            "shoot": is_shooting,
            "start_evil": i == 0,
            "start_cry": i == EVIL_AT_TARGET,
            "start_shoot": i == EVIL_AT_TARGET,
            "end_shoot": i == num_months * FRAMES_PER_MONTH,
        }

    start_date = datetime(1991, 12, 1)
    cur_date = start_date

    max_iters_months = diff_month(NOW, start_date)
    max_iters = max_iters_months * FRAMES_PER_MONTH

    audio_len_ms = calculate_ms(max_iters_months)
    base_audio = AudioSegment.silent(duration=audio_len_ms)

    dur_0_mon = diff_month(datetime(epochs[1], 1, 1), start_date)
    dur_last_mon = diff_month(NOW, datetime(epochs[-1], 1, 1))
    dur_0_ms = calculate_ms(dur_0_mon)
    dur_last_ms = calculate_ms(dur_last_mon)

    combined = epoch_sounds[0][: dur_0_ms + AUDIO_CROSSFADE_DURAION // 2]
    for ie in range(1, len(epochs) - 1):
        dur_e_mon = diff_month(datetime(epochs[ie + 1], 1, 1), datetime(epochs[ie], 1, 1))
        dur_e_ms = calculate_ms(dur_e_mon)
        combined = combined.append(epoch_sounds[ie][:dur_e_ms + AUDIO_CROSSFADE_DURAION],
                                   crossfade=AUDIO_CROSSFADE_DURAION)

    combined = combined.append(epoch_sounds[-1][:dur_last_ms + AUDIO_CROSSFADE_DURAION // 2],
                               crossfade=AUDIO_CROSSFADE_DURAION)

    # play(combined)
    combined.export(COMBINED_AUDIO_NAME, format="mp3")
    rest_audio = epoch_sounds[-1][dur_last_ms + AUDIO_CROSSFADE_DURAION // 2:]
    rest_audio.export(REST_AUDIO_NAME, format="mp3")
    base_audio = base_audio.overlay(combined, position=0)

    cut_i_tl, cut_j_tl = None, None
    cut_i_br, cut_j_br = None, None
    background_image = None
    world_changed = False

    with imageio.get_writer(MOVIE_NAME, mode='I', fps=FPS) as writer:
        for i in tqdm(range(-1, max_iters + 1)):
            ax.clear()
            ax.set_facecolor(BACKGROUND_COLOR if i >= 0 else BACKGROUND_COLOR_0)

            cur_conflicts = get_conflict_by_date(cur_date, conflicts)

            if (i < 0) or (i == max_iters):
                cur_conflicts = []

            ax.set_ylim(-50, 80)
            ax.set_xlim(-170, 170)

            if cur_date >= datetime(2022, 9, 1) and not world_changed:
                world_changed = True
                world = deepcopy(world_sep22)

            world.plot(ax=ax, fc=WORLD_COLOR if i >= 0 else WORLD_COLOR_0)

            for conflict in cur_conflicts:
                start_index = diff_month(conflict["start"], start_date) * FRAMES_PER_MONTH

                side_a = conflict["side_a"]
                side_b = conflict["side_b"]

                group_a = None
                if conflict.get("is_nato_a", False):
                    year = cur_date.year
                    group_a = [c for y in LIST_NATO for c in LIST_NATO[y] if y <= year]

                try:
                    params = animate_conflict(i - start_index, conflict["num_months"], side_a, side_b, group_a=group_a)
                    if params["start_evil"]:
                        base_audio = base_audio.overlay(conflict["audio"], position=frames_to_ms(i))

                except Exception as e:
                    print(e)
                    print(side_a)
                    print(side_b)
                    continue

            ax.text(-160, -45, cur_date.strftime("%Y, %B"), style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 0})

            plt.xticks([])
            plt.yticks([])

            fig.canvas.draw()
            mat = np.array(fig.canvas.renderer._renderer)
            if i <= 0:
                print("Test image shape:", mat.shape)
                mask_ii, mask_jj = np.where(mat[..., -1] > 0)
                cut_i_tl = np.min(mask_ii) - VERTICAL_MARGIN
                cut_i_br = np.max(mask_ii) + VERTICAL_MARGIN

                cut_j_tl, cut_j_br = np.min(mask_jj), np.max(mask_jj)
                width = cut_j_br - cut_j_tl
                height = cut_i_br - cut_i_tl
                remainder_x, remainder_y = width % 16, height % 16

                # tighten x, expanding y
                if remainder_x:
                    width -= remainder_x
                    cut_j_tl += remainder_x // 2
                    cut_j_br = cut_j_tl + width
                if remainder_y:
                    height += 16 - remainder_y
                    cut_i_tl -= (16 - remainder_y) // 2
                    cut_i_br = cut_i_tl + height

                print("Cutting coords initial:")
                print(
                    "min_i", np.min(mask_ii),
                    "min_j", np.min(mask_jj),
                    "max_ii", np.max(mask_ii),
                    "max_jj", np.max(mask_jj),
                )

                mat = mat[cut_i_tl: cut_i_br, cut_j_tl: cut_j_br, ...]
                im_test = Image.fromarray(mat)
                im_test.save(f"test_img_{-i}.png")  # trick for negative i
                background_image = Image.new("RGBA", im_test.size, BACKGROUND_COLOR if i >= 0 else BACKGROUND_COLOR_0)

            if i > 0:
                mat = mat[cut_i_tl: cut_i_br, cut_j_tl: cut_j_br, ...]

            im_render = Image.fromarray(mat)

            background_image_mat = background_image.copy()
            background_image_mat.paste(im_render, (0, 0), im_render)

            if i < 0 or i >= max_iters - 1:
                background_image_mat.save(f"zero_image_{-i}.png")

            mat = np.array(background_image_mat)

            if 0 <= i < max_iters:
                writer.append_data(mat)

            if (i + 1) % FRAMES_PER_MONTH == 0 and i >= 0:
                cur_date += relativedelta(months=1)

    base_audio.export(MOVIE_SOUND, format="mp3")

    cmd = f"""ffmpeg -i {MOVIE_NAME} -i {MOVIE_SOUND} -c:v copy -map 0:v -map 1:a -y {OUTPUT_NAME}"""
    subprocess.call(cmd, shell=True)
    print('Muxing Done')


def cumulative_conflicts(folder):
    html = os.path.join(folder, "List of interstate wars since 1945 - Wikipedia.html")
    global ROUND_TO_MONTHS
    ROUND_TO_MONTHS = False
    conflicts = get_conflicts_from_table(html)
    data = []
    for c in conflicts:
        side_a = c["side_a"]
        if isinstance(side_a, str):
            side_a = [side_a]
        start = c["start"]
        finish = c["finish"]
        if c["is_nato_a"]:
            side_a += [cnt for y in LIST_NATO for cnt in LIST_NATO[y] if (y <= start.year) and (cnt not in side_a)]

        cur_data = {
            "side_a": side_a,
            "start": start,
            "finish": finish,
        }

        data.append(cur_data)

    data.sort(key=lambda x: x["start"])
    print("DATA:", data)
    all_sides = set()

    for d in data:
        all_sides.update(d["side_a"])

    side2active = {s: 0 for s in all_sides}

    data = deque(data)

    date = datetime(1991, 12, 23)

    result = None #pd.DataFrame(side2active, index=[date])

    cur_conf = deque([])

    def add_to_curr_conf(item):
        ins_at = 0
        for i, conf in enumerate(cur_conf):
            if item["finish"] >= conf["finish"]:
                break
            ins_at += 1
        cur_conf.insert(ins_at, item)

    def filter_existing(cur_date):
        while cur_conf:
            if cur_conf[-1]["finish"] <= cur_date:
                cur_conf.pop()
            elif cur_conf[-1]["finish"] > cur_date:
                break

    while date <= NOW:

        while data:
            if data[0]["start"] > date:
                break
            elif data[0]["finish"] <= date:
                data.popleft()
            elif (data[0]["start"] <= date) and (data[0]["finish"] > date):
                add_to_curr_conf(data.popleft())
            else:
                print(data[0])
                raise

        # if cur_conf:
        #     print(cur_conf)

        filter_existing(date)

        state = deepcopy(side2active)
        # state["date"] = date
        for conf in cur_conf:
            for side in conf["side_a"]:
                state[side] += 1

        cur_df = pd.DataFrame(state, index=[date])
        if result is None:
            result = cur_df
        else:
            result = pd.concat([result, cur_df], ignore_index=False)

        date += timedelta(days=1)

    # print(data)
    # print(cur_conf)
    # print(result.head())

    ts = result.cumsum()
    ts = ts.sort_values(ts.last_valid_index(), axis=1, ascending=False)

    print(ts.head())

    fig, ax = plt.subplots(dpi=150)

    def forward(x):
        return (x - 1000) ** (1 / 2)

    def inverse(x):
        return (x - 1000) ** 2

    ax.set_yscale('linear')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    AX_COL = "white"
    #ax.set_yscale('function', functions=(forward, inverse))
    # plt.figure()
    import matplotlib
    #ts.plot(ax=ax, legend=False)
    cmap = matplotlib.cm.get_cmap('plasma')
    columns = ts.columns.tolist()
    for i, col in enumerate(columns[::-1]):
        color = "red"
        color = cmap(ts[col].values[-1] / ts[columns[0]].values[-1])
        ts[col].plot(ax=ax, legend=False, color=color)
        ax.annotate(xy=(ts.index[-1], ts[col].values[-1]),
                    xytext=(1 + (4 * 5 * (i % 2)), 0),
                    textcoords='offset points',
                    text=col, va='center', color=color)

    ax.yaxis.label.set_color(AX_COL)
    ax.xaxis.label.set_color(AX_COL)
    ax.spines['bottom'].set_color(AX_COL)
    ax.spines['top'].set_color(AX_COL)
    ax.spines['right'].set_color(AX_COL)
    ax.spines['left'].set_color(AX_COL)
    ax.set_xlabel("Date")
    ax.set_ylabel("#Days")
    ax.tick_params(axis='x', colors=AX_COL)
    ax.tick_params(axis='y', colors=AX_COL)
    ax.title.set_color("green")

    plt.title('Cumulative interstate military aggression during my life\naccording to: "wiki / List of interstate wars since 1945"*')

    #ax.set_yscale('log')
    plt.savefig("cumsum.png")
    plt.show()


def portrait(folder):
    rus_dbf = os.path.join(folder, "rus_adm_gadm_2022_v03/rus_admbnda_adm2_gadm_2022_v02.dbf")
    rus_a = gpd.read_file(rus_dbf)

    ua = gpd.read_file(os.path.join(folder, "ukr_admbnda_sspe_20230201_SHP/ukr_admbnda_adm1_sspe_20230201.dbf"))

    NEW_REGIONS_TEMPLATE = {
        "ADM0_EN": "Russian Federation",
        "ADM0_PCODE": "RU",
        "ADM1_EN": "South federal district",
        "ADM1_PCODE": "RU006",
        "ADM2_EN": "",
        "ADM2_PCODE": "",
        "UNHCR_code": "",
        "geometry": None
    }

    new_regions = [
        "Autonomous Republic of Crimea",
        "Donetska",
        "Khersonska",
        "Luhanska",
        "Zaporizka",
        "Sevastopol",
    ]

    appended_regions = []
    for region in new_regions:
        ua.to_crs(rus_a.crs,inplace=True)
        geometry = ua[ua["ADM1_EN"] == region]["geometry"].values[0]
        print("#########",type(geometry))
        print(dir(geometry))
        template = deepcopy(NEW_REGIONS_TEMPLATE)
        template["geometry"] = geometry
        template["ADM2_EN"] = region
        appended_regions.append(template)
    print(ua.head())
    print(rus_a.head())
    print(appended_regions)
    new_regions_df = gpd.GeoDataFrame(appended_regions)
    new_regions_df.set_geometry("geometry")
    new_regions_df.geometry = new_regions_df.geometry.set_crs(4326)
    new_regions_df.geometry = new_regions_df.geometry.to_crs("EPSG:4326")

    print(new_regions_df.head())
    rus = pd.concat([rus_a, new_regions_df])

    from shapely.geometry import LineString
    from shapely.ops import split
    from shapely.affinity import translate

    def shift_geom(shift, gdataframe, plotQ=False):
        # this code is adapted from answer found in SO
        # will be credited here: ???
        shift -= 180
        moved_geom = []
        splitted_geom = []
        border = LineString([(shift, 90), (shift, -90)])

        for row in gdataframe["geometry"]:
            splitted_geom.append(split(row, border))
        for element in splitted_geom:
            items = list(element.geoms)
            for item in items:
                minx, miny, maxx, maxy = item.bounds
                if minx >= shift:
                    moved_geom.append(translate(item, xoff=-180 - shift))
                else:
                    moved_geom.append(translate(item, xoff=180 - shift))

        # got `moved_geom` as the moved geometry
        moved_geom_gdf = gpd.GeoDataFrame({"geometry": moved_geom})

        # can change crs here
        if plotQ:
            fig1, ax1 = plt.subplots(figsize=[8, 6])
            moved_geom_gdf.plot(ax=ax1)
            plt.show()

        return moved_geom_gdf

    # print(rus.head(20))
    chu = rus[rus["ADM2_EN"]=="Chukot"]
    new_rus = shift_geom(90, chu, False)
    # restore the geometry to original geo-location
    # ... geometry now in 1 piece
    # ... option True --> make a plot
    chu = shift_geom(-90, new_rus, False)
    print("CHU")
    print(chu.head())
    from shapely.ops import unary_union
    cu = unary_union(chu["geometry"].values)
    print(type(cu))
    rus.loc[rus["ADM2_EN"] == "Chukot", "geometry"] = cu#cascaded_union(chu["geometry"].values)

    SEP = ";"
    data_file = os.path.join(folder, "data-30-structure-5_add.csv")
    map_json = os.path.join(folder, "russia-cities-master/russia-regions.json")
    with open(map_json, "r") as f:
        regnames = json.load(f)

    with open(data_file, "r", encoding="windows-1251") as f:
        data = [l.strip() for l in f.readlines()]
        header = data[1].split(SEP)
        data = data[2:]
    data = [dict(zip(header, d.split(SEP))) for d in data]
    data = pd.DataFrame(data)

    print(rus_a.head())
    print(data["Религия"].unique())

    rus_subjects = rus_a["ADM2_EN"].unique()
    rus_subjects = sorted(rus_subjects)
    rus_subj_mapping = {rs: rs.strip().replace("'", "").lower() for rs in rus_subjects}
    inverse_mapping = {v: k for k, v in rus_subj_mapping.items()}
    print(inverse_mapping)
    # print(regnames[0])

    en2rus = {rn["label"]: rn["name"] for rn in regnames}
    en2rusfull = {rn["label"]: rn["fullname"] for rn in regnames}
    print(en2rus)

    matched, unmatched = [], []

    en2rus2geoitem = dict()

    # inv mapping needed for geo_db
    for item in inverse_mapping:
        if item not in en2rus:
            # USEFUL PRINT: print("###", item, "CLOSEST", get_close_matches(item, list(en2rus.keys())))
            closest_matches = get_close_matches(item, list(en2rus.keys()))
            if closest_matches:
                match = closest_matches[0]
            else:
                match = None
        else:
            match = item

        if item == "maga buryatdan":
            match = None
            # print("MM", match)

        if match is not None:
            en2rus2geoitem[match] = inverse_mapping[item] #item
            matched.append(item)
        else:
            unmatched.append(item)

    print("UNM")
    print(unmatched)
    print("e2r2g")
    print(en2rus2geoitem)
    #for item in matched:
    #    inverse_mapping.pop(item)
    #print(inverse_mapping)
    en2rus_na = deepcopy(en2rus)
    for item in en2rus2geoitem:
        en2rus_na.pop(item)

    print("NA")
    print(en2rus_na)
    print("MANUAL_CORRECTIONS")
    MANUAL_MAP = {
        "altajskij": "gorno-altay",
        "kaluzhskaya": "kaluga",
        "hmao": "khanty-mansiy",
        "yanao": "nenets",
        "severnaya_osetiya": "north ossetia",
        "orlovskaya": "orel",
        "yakutiya": "sakha",
        "neneckij": "yamal-nenets",
        "magadanskaya": "maga buryatdan",
    }

    for item in MANUAL_MAP:
        en2rus2geoitem[item] = inverse_mapping[MANUAL_MAP[item]]

    print("EN2RUS2GEOITEM")
    print(json.dumps(en2rus2geoitem, indent=2))

    region_entries = data["Субъект"].unique()
    print(region_entries)
    en2rus2datareg = dict()
    for item in en2rusfull:
        rusfull = en2rusfull[item]
        match = get_close_matches(rusfull, region_entries)
        print("RF", rusfull, "MATCH", match)
        en2rus2datareg[item] = match[0]

    print(json.dumps(en2rus2datareg, indent=2, ensure_ascii=False))
    unmatched_regs = []
    for entry in region_entries:
        if entry not in en2rus2datareg.values():
            unmatched_regs.append(entry)

    print("Not found")
    print(unmatched_regs)

    religions = data["Религия"].unique()
    for religion in religions:
        rus[religion] = 0

    print(rus.head())
    print(data.head())
    print(en2rus2datareg)
    print(en2rus2geoitem)

    datareg2en = {v: k for k, v in en2rus2datareg.items()}

    rus["region"] = rus["ADM2_EN"]

    print(religions)

    ADDED_REGIONS_EN = {
        "Донецкая Народная Республика": "Donetska",
        "Херсонская область": "Khersonska",
        "Луганская Народная Республика": "Luhanska",
        "Запорожская область": "Zaporizka",
        "г. Севастополь": "Sevastopol",
        "Автономная Республика Крым": "Autonomous Republic of Crimea",
    }

    for idx, row in data.iterrows():
        religion = row["Религия"]
        value = row["Значение"]
        reg = row["Субъект"]
        if reg not in datareg2en:
            print("Missed Region:", reg)
            target_reg = ADDED_REGIONS_EN[reg]
        else:
            en_reg = datareg2en[reg]
            target_reg = en2rus2geoitem[en_reg]
        rus.loc[rus["ADM2_EN"] == target_reg, religion] = float(value.replace(",", "."))
        rus.loc[rus["ADM2_EN"] == target_reg, "region"] = reg

    print(rus.head(89))

    # rus = rus.to_crs(epsg=3576)

    fig, ax = plt.subplots(dpi=300, frameon=False)

    def plotCountryPatch(axes, reg_name, religion, fcolor):
        # plot a country on the provided axes
        nami = rus[rus["region"] == reg_name]
        alpha = nami[religion].item() / 100

        if alpha == 0:
            return

        namigm = nami.__geo_interface__['features']  # geopandas's geo_interface
        namig0 = {'type': namigm[0]['geometry']['type'], \
                  'coordinates': namigm[0]['geometry']['coordinates']}
        axes.add_patch(PolygonPatch(namig0, fc=fcolor, alpha=alpha,
                                    edgecolor=EDGE_COLOR, zorder=2,
                                    linestyle=''))

    cur_religion = "Православие"
    #
    LISTED_RELIGIONS = ["Православие", "Своя вера", "Атеизм", "Буддизм", "Язычество", "Ислам", "Католицизм", "Старообрядчество", "Прочие"] #"Своя вера", "Атеизм",
    result_images = []
    plt.axis("off")
    plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False,
                    labelright=False, labelbottom=False)
    ax.spines['top'].set_visible(False)  # ax.spines[:].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    BG_COLOR = "#000000ff"
    W, H = 1504, 728
    left, upper = 230, 356
    PATCH_COLOR = "#ed9f16" #"#ff0000"

    X_MARGIN = W // 10

    INTERMEDIATE_IMG_SIZE = H // 7

    color_bar_img = Image.new("RGBA", (W, H // 7), BG_COLOR)
    bg_img = color_bar_img.copy()
    patch_img = Image.new("RGBA", color_bar_img.size, PATCH_COLOR)
    color_bar = np.asarray(color_bar_img.copy()).copy()
    bar_len = W - 2 * X_MARGIN
    bar = np.linspace(255, 0, bar_len, dtype=np.uint8)
    y_pos = color_bar.shape[0] // 2
    color_bar[: y_pos, X_MARGIN: X_MARGIN + bar_len, -1] = bar
    color_bar_img = Image.fromarray(color_bar, "RGBA")
    color_bar_img = Image.alpha_composite(patch_img, color_bar_img)
    from utils.visualize import get_font_size_for_target_size, draw_text_image
    text_size = (X_MARGIN, (color_bar.shape[0] - y_pos) * 9 // 10)
    text_100, font_size = draw_text_image(text_size, "100%", "utils/fonts/sagrada/SAGRADA.otf",
                                          background_color=BG_COLOR, side_margin=50, return_font_size=True,
                                          anchor="ls")
    text_0, font_size = draw_text_image(text_size, "0%", "utils/fonts/sagrada/SAGRADA.otf",
                                          background_color=BG_COLOR, side_margin=50, return_font_size=True,
                                          anchor="ls", font_size=font_size)
    text_margin = y_pos // 10
    draw = ImageDraw.Draw(color_bar_img)
    CYRILLIC_FONT = "utils/fonts/sagrada/SAGRADA.otf"
    fnt = ImageFont.truetype(CYRILLIC_FONT, font_size)
    draw.text((W-X_MARGIN, y_pos + text_margin), "100", font=fnt, fill="#ffffffff", anchor="rt")
    draw.text((X_MARGIN, y_pos + text_margin), "0", font=fnt, fill="#ffffffff", anchor="lt")
    draw.text((W // 2, y_pos + text_margin), "% НАСЕЛЕНИЯ РЕГИОНА", font=fnt, fill="#ffffffff", anchor="mt")
    #color_bar_img.paste(text_100, (W - X_MARGIN - text_100.size[0] // 2, y_pos), text_100)
    #color_bar_img.paste(text_0, (X_MARGIN - text_0.size[0] // 2, y_pos), text_0)
    # color_bar_img.show()

    inter_img_religion = Image.new("RGBA", (W, INTERMEDIATE_IMG_SIZE), BG_COLOR)

    symbols_margin = INTERMEDIATE_IMG_SIZE // 10
    multiply_symbol = Image.new("RGBA", (INTERMEDIATE_IMG_SIZE-symbols_margin, INTERMEDIATE_IMG_SIZE-symbols_margin), BG_COLOR)
    dr = ImageDraw.Draw(multiply_symbol)
    dr.line((0, 0) + multiply_symbol.size, fill="#ffffff", width=8)
    dr.line((0, *multiply_symbol.size[::-1], 0), fill="#ffffff", width=8)
    base_img = Image.new("RGBA", (INTERMEDIATE_IMG_SIZE, INTERMEDIATE_IMG_SIZE), BG_COLOR)
    base_img.paste(multiply_symbol, (symbols_margin, symbols_margin), multiply_symbol)
    multiply_symbol = base_img

    add_symbol = Image.new("RGBA", (INTERMEDIATE_IMG_SIZE-symbols_margin, INTERMEDIATE_IMG_SIZE-symbols_margin), BG_COLOR)
    dr = ImageDraw.Draw(add_symbol)
    dr.line((add_symbol.size[0] // 2, 0, add_symbol.size[0] // 2, add_symbol.size[1]), fill="#ffffff", width=8)
    dr.line((0, add_symbol.size[1] // 2, add_symbol.size[0], add_symbol.size[1] // 2,), fill="#ffffff", width=8)
    base_img = Image.new("RGBA", (INTERMEDIATE_IMG_SIZE, INTERMEDIATE_IMG_SIZE), BG_COLOR)
    base_img.paste(add_symbol, (symbols_margin, symbols_margin), add_symbol)
    add_symbol = base_img

    equal_symbol = Image.new("RGBA", (INTERMEDIATE_IMG_SIZE, INTERMEDIATE_IMG_SIZE), BG_COLOR)
    dr = ImageDraw.Draw(equal_symbol)
    dr.line((0, equal_symbol.size[1] // 3, equal_symbol.size[0], equal_symbol.size[1] // 3,), fill="#ffffff", width=8)
    dr.line((0, 2 * equal_symbol.size[1] // 3, equal_symbol.size[0], 2 * equal_symbol.size[1] // 3,), fill="#ffffff", width=8)

    def x_middle_paste(base, overlay, y_pos):
        overlay_w = overlay.size[0]
        base_w = base.size[0]
        x_pos = (base_w - overlay_w) // 2
        base.paste(overlay, (x_pos, y_pos))
        return base

    def y_append(img_a, img_b, bg_color=None):
        wa, ha = img_a.size
        wb, hb = img_b.size
        base = Image.new("RGBA", (max(wa, wb), ha+hb), BG_COLOR if bg_color is None else bg_color)
        base = x_middle_paste(base, img_a, 0)
        base = x_middle_paste(base, img_b, ha)
        return base

    def v_concat_images(images, bg_color=None):
        if len(images) == 0:
            return None
        base = images[0]
        for i in range(1, len(images)):
            base = y_append(base, images[i], bg_color)
        return base

    def make_religion_name_img(religion_name, font_multiplier=2, b_img=None):
        if b_img is None:
            rel_name_img = inter_img_religion.copy()
        else:
            rel_name_img = b_img.copy()
        draw = ImageDraw.Draw(rel_name_img)
        fnt = ImageFont.truetype(CYRILLIC_FONT, font_multiplier * font_size)
        draw.text((rel_name_img.size[0] // 2, 0), religion_name.capitalize(),
                  fill=PATCH_COLOR, font=fnt, anchor="mt")
        return rel_name_img

    def upscale_to_w(img, target_size):
        w, h = img.size
        tw, th = target_size
        scale_x = tw / w
        new_h = scale_x * h
        new_size = (tw, int(round(new_h)))
        img = img.resize(new_size, Image.LANCZOS)
        return img

    color_bar_img.save("colorbar.png")

    ax.clear()
    ax.set_facecolor("#00000000")
    # rus.apply(lambda x: plotCountryPatch(ax, x["region"], cur_religion, PATCH_COLOR), axis=1)
    rus.plot(ax=ax, fc=PATCH_COLOR)

    fig.canvas.draw()
    mat = np.array(fig.canvas.renderer._renderer)
    img = Image.fromarray(mat, "RGBA")
    img = img.crop((left, upper, left + W, upper + H))
    rus_alpha = img.getchannel("A")
    #bg_image = Image.new("RGBA", img.size, color=BG_COLOR)
    #overlay = Image.alpha_composite(bg_image, img)
    #overlay.putalpha(rus_alpha)
    #overlay.show()
    #exit(0)
    TARGET_BIN = os.path.join(folder, "result_religion_images.bin")
    RELEASE_SIZE = (1080, 1920)
    MARGINS = 20

    V_LINE = RELEASE_SIZE[0] // 15

    if os.path.exists(TARGET_BIN):
        textual_frame_iters = 3 * FPS
        textual_fade_iters = FPS // 4

        def animate_textual(num_iters, fade_iters, linear_min_val, linear_max_val, fade_in=True, fade_out=True):
            alpha_anim = np.array([255] * num_iters, dtype=np.uint8)
            if fade_in:
                alpha_anim[:fade_iters] = np.linspace(0, 255, fade_iters, dtype=np.uint8)
            if fade_out:
                alpha_anim[-fade_iters:] = np.linspace(255, 0, fade_iters, dtype=np.uint8)

            v_pos = np.array([linear_max_val] * num_iters, dtype=np.int_)
            v_pos[:num_iters - fade_iters] = np.linspace(linear_min_val, linear_max_val,
                                                         num_iters - fade_iters, dtype=np.int_)

            return alpha_anim, v_pos

        def make_text_video(background_color,
                            three_stripes_colors,
                            video_name,
                            num_iters,
                            fade_iters,
                            linear_min_val,
                            linear_max_val,
                            add_image,
                            texts,
                            use_flag=True,
                            animate_flag=True,
                            fade_in=True,
                            fade_out=True,
                            prev_color=None,
                            ):
            BG_IMG = Image.new("RGBA", RELEASE_SIZE, background_color)
            if prev_color is not None:
                PREV_BG = Image.new("RGBA", RELEASE_SIZE, prev_color)

            assert len(three_stripes_colors) == 3
            alpha_anim, v_pos = animate_textual(num_iters, fade_iters, linear_min_val, linear_max_val, fade_in, fade_out)
            with imageio.get_writer(video_name + ".mp4", mode='I', fps=FPS, macro_block_size=1) as writer:
                for val_alpha, v_p in zip(alpha_anim, v_pos):
                    if not animate_flag:
                        v_p = None
                    FLAG_IMG = Image.new("RGBA", RELEASE_SIZE, background_color if prev_color is None else "#00000000")
                    FLAG_IMG = np.array(FLAG_IMG).copy()
                    FLAG_IMG[:v_p, 6 * V_LINE:7 * V_LINE, :] = np.array(three_stripes_colors[0])
                    FLAG_IMG[:v_p, 7 * V_LINE:8 * V_LINE, :] = np.array(three_stripes_colors[1])
                    FLAG_IMG[:v_p, 8 * V_LINE:9 * V_LINE, :] = np.array(three_stripes_colors[2])
                    FLAG_IMG = Image.fromarray(FLAG_IMG, "RGBA")
                    LETTERS_BASE = Image.new("RGBA", (RELEASE_SIZE[0], RELEASE_SIZE[1] // 12), "#00000000")

                    if add_image is not None:
                        base = add_image.copy()
                        base.thumbnail((RELEASE_SIZE[0] - 2 * MARGINS, RELEASE_SIZE[1] - 2 * MARGINS))
                        base = [base]
                    else:
                        base = []

                    text_ims = [make_religion_name_img(t, b_img=LETTERS_BASE) for t in texts]
                    new_img = v_concat_images(text_ims + base, background_color if prev_color is None else "#00000000")

                    if use_flag:
                        FLAG_IMG.paste(new_img, (0, (RELEASE_SIZE[1] - new_img.size[1]) // 2), new_img)
                        # RU_FLAG_IMG.paste(ru_base, (MARGINS, MARGINS), ru_base)
                        if prev_color is None:
                            FLAG_IMG.putalpha(val_alpha)
                    else:
                        FLAG_IMG = Image.new("RGBA", FLAG_IMG.size, background_color)
                        FLAG_IMG.paste(new_img, (0, (RELEASE_SIZE[1] - new_img.size[1]) // 2), new_img)
                        if prev_color is None:
                            FLAG_IMG.putalpha(val_alpha)

                    background_img = BG_IMG.copy()
                    if prev_color is not None:
                        background_img = Image.blend(PREV_BG.copy(), background_img, val_alpha / 255)
                        background_img.putalpha(255)

                    frame = Image.alpha_composite(background_img, FLAG_IMG)
                    frame = np.array(frame, dtype=np.uint8)
                    writer.append_data(frame)

        text_background_color = "#682828ff"

        make_text_video(background_color=text_background_color,
                        three_stripes_colors=[[255, 255, 255, 255], [0, 33, 118, 255], [201, 58, 51, 255]],
                        video_name="Religion_intro",
                        num_iters=5 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=img,
                        texts=["КАКОВ ЛИК", "РОССИЙСКОЙ ВЕРЫ?"]
                        )

        make_text_video(background_color=text_background_color,
                        three_stripes_colors=[[255, 255, 255, 0], [183, 71, 67, 255], [201, 58, 51, 0]],
                        video_name="Religion_chapter_1",
                        num_iters=4 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=None,
                        texts=["1", " ", "РЕГИОНАЛЬНЫЕ СВЕДЕНИЯ"],
                        use_flag=False
                        )

        make_text_video(background_color=text_background_color,
                        three_stripes_colors=[[255, 255, 255, 0], [183, 71, 67, 255], [201, 58, 51, 0]],
                        video_name="Religion_chapter_2",
                        num_iters=4 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=None,
                        texts=["2", " ", "ОТОБРАЖЕНИЕ ДАННЫХ"],
                        use_flag=False
                        )

        make_text_video(background_color=text_background_color,
                        three_stripes_colors=[[255, 255, 255, 0], [183, 71, 67, 255], [201, 58, 51, 0]],
                        video_name="Religion_chapter_3",
                        num_iters=4 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=None,
                        texts=["3", " ", "КОМПОЗИЦИЯ И ВЫВОД"],
                        use_flag=False
                        )

        make_text_video(background_color=text_background_color,
                        three_stripes_colors=[[255, 255, 255, 255], [0, 33, 118, 255], [201, 58, 51, 255]],
                        video_name="Religion_chapter_4",
                        num_iters=4 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=img,
                        texts=["С УВАЖЕНИЕМ", "К ВАШЕЙ ВЕРЕ!"],
                        use_flag=True,
                        animate_flag=False
                        )

        with open(TARGET_BIN, "rb") as frb:
            res_data = pickle.load(frb)
        result = res_data.pop("result")

        make_text_video(background_color="#00000000",
                        three_stripes_colors=[[255, 255, 255, 255], [0, 33, 118, 255], [201, 58, 51, 255]],
                        video_name="Religion_chapter_5",
                        num_iters=3 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=result["rus_alpha"],
                        texts=["ЛИК РОССИЙСКОЙ ВЕРЫ"],
                        use_flag=False,
                        animate_flag=False,
                        fade_out=False
                        )

        make_text_video(background_color="#00000000",
                        three_stripes_colors=[[255, 255, 255, 255], [0, 33, 118, 255], [201, 58, 51, 255]],
                        video_name="Religion_chapter_6",
                        num_iters=4 * FPS // 2,
                        fade_iters=FPS // 4,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=result["rus_alpha"],
                        texts=["ЛИК РОССИЙСКОЙ ВЕРЫ"],
                        use_flag=True,
                        animate_flag=True,
                        fade_out=False,
                        fade_in=False,
                        )

        make_text_video(background_color=text_background_color,
                        three_stripes_colors=[[255, 255, 255, 255], [0, 33, 118, 255], [201, 58, 51, 255]],
                        video_name="Religion_chapter_7",
                        num_iters=3 * FPS // 2,
                        fade_iters=FPS // 2,
                        linear_min_val=0,
                        linear_max_val=RELEASE_SIZE[1],
                        add_image=result["rus_alpha"],
                        texts=["ЛИК РОССИЙСКОЙ ВЕРЫ"],
                        use_flag=True,
                        animate_flag=False,
                        fade_in=True,
                        fade_out=False,
                        prev_color="#000000ff"
                        )

        num_religions = len(list(res_data.keys()))
        # 9 Religions
        res_data = dict(sorted(res_data.items()))
        others = res_data.pop("Прочие")
        res_data["Прочие"] = others

        fade_in_iters = FPS // 2
        total_iters = 3 * FPS

        part_3_queue = []
        religions_list = sorted(LISTED_RELIGIONS)
        _others = religions_list.pop(religions_list.index("Прочие"))
        religions_list.append(_others)

        rel_mapping = {
            "Православие": "Прав.",
            "Ислам": "Исл.",
            "Буддизм": "Буд.",
            "Старообрядчество": "Стар.",
            "Атеизм": "Ате.",
            "Своя вера": "Своя",
            "Прочие": "Прч.",
            "Католицизм": "Кат.",
            "Язычество": "Языч.",
        }

        base_religions_dict = {lr: 0 for lr in religions_list}
        # base_religions_dict = {**{"Субъект": ""}, **base_religions_dict}
        all_analytica = []
        data_analytica = []
        j=0
        all_subjects = []
        for subject_name, group in data.groupby("Субъект"):
            cur_dict = deepcopy(base_religions_dict)
            for row_index, row in group.iterrows():
                cur_dict[row["Религия"]] = round(float(row["Значение"]), ndigits=1)
            # cur_dict["Субъект"] = subject_name
            subject_name = subject_name.replace("область", "обл.")
            subject_name = subject_name.replace("Республика", "Респ.")
            subject_name = subject_name.replace("Край", "Кр.")
            subject_name = subject_name.replace("Автономная", "Авт.")
            subject_name = subject_name.replace("автономная", "авт.")
            subject_name = subject_name.replace("Народная", "Нар.")
            subject_name = subject_name.replace("автономный округ", "АО")
            cur_data = [{"Субъект": subject_name + " " * (28 - len(subject_name)), "Религия": rel_mapping[rel], "Значение": val} for rel, val in cur_dict.items()]
            data_analytica += cur_data
            all_subjects.append(subject_name)

            j += 1
            if j % 23 == 0:
                all_analytica.append(pd.DataFrame(data_analytica))
                data_analytica = []

        all_analytica.append(pd.DataFrame(data_analytica))

        max_size = max([len(sn) for sn in all_subjects])

        data_frame_iters = 2 * FPS
        data_fade_iters = FPS // 4
        alpha_anim = np.array([255] * data_frame_iters, dtype=np.uint8)
        alpha_anim[:data_fade_iters] = np.linspace(0, 255, data_fade_iters, dtype=np.uint8)
        alpha_anim[-data_fade_iters:] = np.linspace(255, 0, data_fade_iters, dtype=np.uint8)
        index_anim = np.arange(len(all_analytica), dtype=np.int32)
        index_anim = np.repeat(index_anim, data_frame_iters)
        alpha_anim = np.tile(alpha_anim, len(all_analytica))

        release_img = Image.new("RGBA", RELEASE_SIZE, text_background_color)
        with imageio.get_writer("Religions_part_1.mp4", mode='I', fps=FPS, macro_block_size=1) as writer:
            for i, alpha_val in zip(index_anim, alpha_anim):
                res_analytica = all_analytica[i].pivot(index="Субъект", columns="Религия", values="Значение")
                sns.set(rc={'axes.facecolor': PATCH_COLOR, 'figure.facecolor': PATCH_COLOR})
                ax.clear()
                fig, ax = plt.subplots(figsize=((len(res_analytica.index) + 1), 14)) #dpi=400
                plt.gcf().set_size_inches(16 // 2, (len(res_analytica.index) + 1) // 2)

                sns.heatmap(res_analytica, ax=ax, annot=True, fmt='g',
                            cmap="YlOrRd", cbar=False, square=True, linecolor="black")
                ax.set_facecolor(BG_COLOR)
                ax.set_xlabel(None)
                ax.set_ylabel(None)
                ax.xaxis.tick_top()  # x axis on top
                ax.xaxis.set_label_position('top')
                ax.spines[:].set_visible(False)
                plt.legend([], [], frameon=False)
                fig.tight_layout()
                fig.subplots_adjust(top=0.95, right=1.05)
                fig.canvas.draw()
                mat = np.array(fig.canvas.renderer._renderer)
                ig = Image.fromarray(mat, "RGBA").convert("RGBA").copy()
                ig = upscale_to_w(ig, (RELEASE_SIZE[0] - 2 * MARGINS, RELEASE_SIZE[1] - 2 * MARGINS))
                ig.putalpha(alpha_val)
                frame = release_img.copy()
                frame.paste(ig, (MARGINS, MARGINS), ig)
                frame = np.array(frame, dtype=np.uint8)
                writer.append_data(frame)
                plt.cla()
                plt.close(fig)

        with imageio.get_writer("Religions_part_2.mp4", mode='I', fps=FPS, macro_block_size=1) as writer:
            for cur_religion, images in res_data.items():
                print(cur_religion)
                cur_frame_a = v_concat_images(images[:-1])
                cur_frame_b = v_concat_images(images[:-2] + images[-1:])
                part_3_queue.append(images[-1].copy())
                frames_per_intro = int(round(fade_in_iters / (len(images) - 2)))

                target_size = cur_frame_a.size
                back_img = Image.new("RGBA", target_size, BG_COLOR)
                release_img = Image.new("RGBA", RELEASE_SIZE, BG_COLOR)
                for i in range(total_iters):
                    if i < fade_in_iters:
                        total_images = i // frames_per_intro
                        cur_frame = v_concat_images(images[:total_images])
                        cur_base = back_img.copy()
                        if cur_frame is not None:
                            cur_base.paste(cur_frame, (0, 0), cur_frame)
                        cur_base.thumbnail((RELEASE_SIZE[0]-MARGINS, RELEASE_SIZE[1]-MARGINS), Image.LANCZOS)
                        release_img.paste(cur_base, (MARGINS, MARGINS))
                    elif i < total_iters // 2:
                        cur_base = cur_frame_a.copy()
                        cur_base.thumbnail((RELEASE_SIZE[0] - MARGINS, RELEASE_SIZE[1] - MARGINS), Image.LANCZOS)
                        release_img.paste(cur_base, (MARGINS, MARGINS))
                    else:
                        cur_base = cur_frame_b.copy()
                        cur_base.thumbnail((RELEASE_SIZE[0] - MARGINS, RELEASE_SIZE[1] - MARGINS), Image.LANCZOS)
                        release_img.paste(cur_base, (MARGINS, MARGINS))

                    frame = np.array(release_img, dtype=np.uint8)
                    writer.append_data(frame)

        exit(0)

        with imageio.get_writer("Religions_part_3.mp4", mode='I', fps=FPS, macro_block_size=1) as writer:
            num_religions_per_frame = 3
            for i in range(0, num_religions, num_religions_per_frame):
                cur_rel_images = part_3_queue[i: i + num_religions_per_frame]
                cur_queue = [add_symbol] * (num_religions // num_religions_per_frame - 1)
                if i // num_religions_per_frame == num_religions // num_religions_per_frame - 1:
                    cur_queue.append(equal_symbol)
                else:
                    cur_queue.append(add_symbol)
                res_queue = [None] * (len(cur_queue) + len(cur_rel_images))
                print(i, len(res_queue))
                res_queue[::2] = cur_rel_images
                res_queue[1::2] = cur_queue
                cur_frame_a = v_concat_images(res_queue)
                frames_per_intro = int(round(fade_in_iters / (len(res_queue) - 1)))
                target_size = cur_frame_a.size
                back_img = Image.new("RGBA", target_size, BG_COLOR)
                release_img = Image.new("RGBA", RELEASE_SIZE, BG_COLOR)
                for j in range(total_iters):
                    if j < fade_in_iters:
                        total_images = j // frames_per_intro
                        cur_frame = v_concat_images(res_queue[:total_images])
                        cur_base = back_img.copy()
                        if cur_frame is not None:
                            cur_base.paste(cur_frame, (0, 0), cur_frame)
                        cur_base.thumbnail((RELEASE_SIZE[0]-MARGINS, RELEASE_SIZE[1]-MARGINS), Image.LANCZOS)
                        release_img.paste(cur_base, (MARGINS, MARGINS))
                    else:
                        cur_base = cur_frame_a.copy()
                        cur_base.thumbnail((RELEASE_SIZE[0] - MARGINS, RELEASE_SIZE[1] - MARGINS), Image.LANCZOS)
                        release_img.paste(cur_base, (MARGINS, MARGINS))

                    frame = np.array(release_img, dtype=np.uint8)
                    writer.append_data(frame)

            # release_img = Image.new("RGBA", RELEASE_SIZE, BG_COLOR)
            # result_img = result["overlay"]
            # name_img = make_religion_name_img("ЛИК РОССИЙСКОЙ ВЕРЫ")
            # frame = v_concat_images([name_img, result_img])
            # frame.thumbnail((RELEASE_SIZE[0] - MARGINS, RELEASE_SIZE[1] - MARGINS), Image.LANCZOS)
            # release_img.paste(frame, (MARGINS, (RELEASE_SIZE[1] - frame.size[1]) // 2))
            # frame = np.array(release_img, dtype=np.uint8)
            # for k in range(2 * total_iters):
            #     writer.append_data(frame)

        #print(result)
        #print(len(list(res_data.keys())))
        return 

    res_data = dict()

    for i, cur_religion in tqdm(enumerate(LISTED_RELIGIONS)):
        ax.clear()
        ax.set_facecolor("#00000000")
        rus.apply(lambda x: plotCountryPatch(ax, x["region"], cur_religion, PATCH_COLOR), axis=1)
        rus.plot(ax=ax, fc="#ffffff00")

        fig.canvas.draw()
        mat = np.array(fig.canvas.renderer._renderer)
        img = Image.fromarray(mat, "RGBA")
        cur_alpha = img.getchannel("A")

        cur_religion_img = glob(os.path.join(folder, "religions", cur_religion + "*"))[0]
        cur_religion_img = Image.open(cur_religion_img)

        if i == 0:
            cur_religion_img.thumbnail(img.size, Image.LANCZOS)
            cur_religion_img = cur_religion_img.resize(img.size, Image.LANCZOS)
            cur_religion_img.putalpha(255)
        else:
            back_img = Image.new("RGBA", img.size)
            # cur_religion_img.thumbnail((W, H), Image.LANCZOS)
            cur_religion_img = cur_religion_img.resize((W, H), Image.LANCZOS)
            cur_religion_img.putalpha(255)
            smaller_copy = cur_religion_img.copy()
            smaller_copy.save(cur_religion + "_resized.png")
            back_img.paste(cur_religion_img, (left, upper))
            cur_religion_img = back_img.copy()

        def save_img(base_copy, postfix, alpha=None):

            if alpha is not None:
                base_copy.putalpha(alpha)
                # after GIMP preview

            unclustered_religion_img = base_copy.crop((left, upper, left + W, upper + H))
            ###
            unclustered_religion_img.save(cur_religion + f"_{postfix}.png")

            bg_image = Image.new("RGBA", unclustered_religion_img.size, color=BG_COLOR)
            overlay = Image.alpha_composite(bg_image, unclustered_religion_img)
            overlay.save(cur_religion + f"_{postfix}_overlay.png")
            overlay_rualpha = overlay.copy()
            overlay_rualpha.putalpha(rus_alpha)
            overlay_rualpha.save(cur_religion + f"_{postfix}_rus_alpha.png")
            return overlay, overlay_rualpha

        rel_img_nc, _ = save_img(cur_religion_img.copy(), "not_clustered", alpha=cur_alpha.copy())
        part_2_img, _ = save_img(cur_religion_img.copy(), "not_clustered_p2", alpha=None)
        part_2_img.putalpha(255)
        data_img, data_rualpha_img = save_img(img.copy(), "data")

        if i >= 3:
            anp = np.asarray(cur_alpha)
            anp_c = anp.copy()
            anp_c[anp_c < 12] = 0
            anp_c[anp_c != 0] = 255
            contours, hierarchy = cv2.findContours(anp_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(anp_c, contours, -1, (255,), thickness=cv2.FILLED)
            #

            for contour in contours:
                smaller_img = smaller_copy.copy()
                rect = cv2.minAreaRect(contour)

                ((center_x, center_y), (dim_x, dim_y), angle) = rect
                #target_points = cv2.boxPoints(rect)
                #cv2.drawContours(anp_c, [np.intp(target_points)], -1, (255, ), 2)
                #for i, p in enumerate(target_points):
                #    anp_c = cv2.putText(anp_c, f"{i}. {int(angle)}", np.intp(p), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255,), thickness=2, lineType=cv2.LINE_AA)
                # print("DXDY", dim_x, dim_y)
                dim_x = int(round(dim_x))
                dim_y = int(round(dim_y))
                if dim_x <= 1 or dim_y <= 1 or (dim_y * dim_x < 49):
                    continue
                if abs(angle) < 60:
                    smaller_img = smaller_img.resize((dim_x, dim_y), Image.LANCZOS)
                    smaller_img = smaller_img.rotate(360-angle, Image.BICUBIC, expand=True, fillcolor=(255, 255, 255, 0))
                else:
                    smaller_img = smaller_img.resize((dim_y, dim_x), Image.LANCZOS)
                    smaller_img = smaller_img.rotate(360-(90-angle), Image.BICUBIC, expand=True, fillcolor=(255, 255, 255, 0))
                smaller_img_w, smaller_img_h = smaller_img.size
                #print(smaller_img.size, "H", h, "W", w)
                center_x = int(round(center_x))
                center_y = int(round(center_y))
                insert_at_x = center_x - smaller_img_w // 2
                insert_at_y = center_y - smaller_img_h // 2
                smaller_img_mask = anp_c[insert_at_y: insert_at_y + smaller_img_h, insert_at_x: insert_at_x + smaller_img_w].copy()

                smaller_img_mask[smaller_img_mask != 0] = 1
                c_alpha = smaller_img.getchannel("A").copy()
                c_alpha = np.array(c_alpha).copy()
                c_alpha = smaller_img_mask * c_alpha
                c_alpha = np.uint8(c_alpha).copy()
                c_alpha = Image.fromarray(c_alpha, "L")
                cur_religion_img.paste(smaller_img, (insert_at_x, insert_at_y), c_alpha)

            #cv2.imwrite(f"Cont_{cur_religion}.png", anp_c)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #exit(0)

        cur_religion_img.putalpha(cur_alpha)
        # after GIMP preview

        cur_religion_img = cur_religion_img.crop((left, upper, left + W, upper + H))
        ###
        cur_religion_img.save(cur_religion + ".png")
        result_images.append(cur_religion_img.copy())
        bg_image = Image.new("RGBA", cur_religion_img.size, color=BG_COLOR)
        overlay = Image.alpha_composite(bg_image, cur_religion_img)
        overlay.save(cur_religion + "_overlay.png")
        overlay_rualpha = overlay.copy()
        overlay_rualpha.putalpha(rus_alpha)
        overlay_rualpha.save(cur_religion + f"_overlay_rus_alpha.png")

        before_img = cur_religion_img.copy()
        before_img.putalpha(255)
        before_img.save(cur_religion + "_before.png")

        religion_name_img = make_religion_name_img(cur_religion)
        ordered_images = [religion_name_img, part_2_img, multiply_symbol, data_img, color_bar_img,
                          equal_symbol, rel_img_nc, overlay]
        res_data[cur_religion] = ordered_images

    base_img = result_images[0]
    result_images = deque(result_images[1:])
    while result_images:
        cur_img = result_images.popleft()
        base_img = Image.alpha_composite(base_img, cur_img)
        print("<<POP>>")

    res_data["result"] = dict()

    base_img.save("religions_result_transparent.png")
    res_data["result"]["transparent"] = base_img
    bg_image = Image.new("RGBA", base_img.size, color=BG_COLOR)
    overlay = Image.alpha_composite(bg_image, base_img)
    overlay.save("religions_result_overlay.png")
    res_data["result"]["overlay"] = overlay
    overlay.putalpha(rus_alpha)
    overlay.save("religions_result_overlay_rus_alpha.png")
    res_data["result"]["rus_alpha"] = overlay

    with open(TARGET_BIN, "wb") as fwb:
        pickle.dump(res_data, fwb)


if __name__ == "__main__":
    portrait(sys.argv[1])
    # cumulative_conflicts(sys.argv[1])
    # main(sys.argv[1])

