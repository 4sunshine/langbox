from datetime import timedelta, datetime
import sys

import numpy as np
from PIL import Image
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


FRAMES_PER_MONTH = 3 #-- TODO: 3 for release!!!!!
FPS = 30
EVIL_AT_TARGET_MONTHS = 1
EVIL_AT_TARGET = EVIL_AT_TARGET_MONTHS * FRAMES_PER_MONTH
SECONDS_PER_MONTH = round(FRAMES_PER_MONTH / FPS, 1)
SEED = 15266334
random.seed(SEED)
VERTICAL_MARGIN = 40
LEFT_OFFSET = 1
BACKGROUND_COLOR = "#aaffbb"
WORLD_COLOR = "#006633"

DURATIONS_TO_FORGET = 3
MIN_MONTHS_FORGET = 36
MAX_MONTHS_FORGET = 60

AUDIO_CROSSFADE_DURAION = 1000  # ms

EVIL_COLOR = '#0a0909' #'#21618c'#'navy'
SAD_COLOR = '#ff2f2f'
EDGE_COLOR = '#ff2f2f'
ALPHA_COUNTRY = 0.9

NOW = datetime.now()


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
    "Tajikistan": "TJK"
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


def get_conflict_data(start, finish, side_a, side_b, index):
    start = parser.parse(start)
    finish = parser.parse(finish)

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
        "is_nato_a": False,
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
        if "nato" in offense.lower():
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

        data = get_conflict_data(start, finish, side_a, side_b, index)

        if "KOS" == side_a:
            is_nato_a = True

        data["is_nato_a"] = is_nato_a

        conflicts.append(data)

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
    world_sep22 = world_sep_2022(world, ua_dbf)

    fig, ax = plt.subplots(dpi=300, frameon=False)
    ax.set_facecolor(BACKGROUND_COLOR)

    html = os.path.join(folder, "List of interstate wars since 1945 - Wikipedia.html")
    conflicts = get_conflicts_from_table(html)

    prepare_audio_fn = partial(make_audio_for_conflict, evil_segment=evil_segment, cry_segment=cry_segment,
                               shoot_is=shoot_is, shoot_modern=shoot_modern, shoot_urban=shoot_urban)

    conflicts = [{**conflict, "audio": prepare_audio_fn(conflict)} for conflict in conflicts]

    def animate_conflict(i, num_months, side_a, side_b, group_a=None):

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
    combined.export("combined.mp3", format="mp3")
    rest_audio = epoch_sounds[-1][dur_last_ms + AUDIO_CROSSFADE_DURAION // 2:]
    rest_audio.export("rest_music.mp3", format="mp3")
    base_audio = base_audio.overlay(combined, position=0)

    cut_i_tl, cut_j_tl = None, None
    cut_i_br, cut_j_br = None, None
    background_image = None

    with imageio.get_writer('movie.mp4', mode='I', fps=FPS) as writer:
        for i in tqdm(range(max_iters)):
            ax.clear()

            cur_conflicts = get_conflict_by_date(cur_date, conflicts)

            ax.set_ylim(-50, 80)
            ax.set_xlim(-170, 170)

            if cur_date < datetime(2022, 9, 1):
                world.plot(ax=ax, fc=WORLD_COLOR)#"#006400")
            else:
                world_sep22.plot(ax=ax, fc=WORLD_COLOR)

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
            if i == 0:
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
                im_test.save("test_img.png")
                background_image = Image.new("RGBA", im_test.size, BACKGROUND_COLOR)

            if i > 0:
                mat = mat[cut_i_tl: cut_i_br, cut_j_tl: cut_j_br, ...]

            im_render = Image.fromarray(mat)

            background_image_mat = background_image.copy()
            background_image_mat.paste(im_render, (0, 0), im_render)
            mat = np.array(background_image_mat)

            writer.append_data(mat)

            if (i + 1) % FRAMES_PER_MONTH == 0:
                cur_date += relativedelta(months=1)
            #
            # if i >= 50:
            #     break

    base_audio.export("movie_sound.mp3", format="mp3")

    cmd = """ffmpeg -i movie.mp4 -i movie_sound.mp3 -c:v copy -map 0:v -map 1:a -y output.mp4"""
    subprocess.call(cmd, shell=True)
    print('Muxing Done')

if __name__ == "__main__":
    main(sys.argv[1])
