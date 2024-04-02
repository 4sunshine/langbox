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


FRAMES_PER_MONTH = 3
FPS = 30
EVIL_AT_TARGET_MONTHS = 1
EVIL_AT_TARGET = EVIL_AT_TARGET_MONTHS * FRAMES_PER_MONTH
SECONDS_PER_MONTH = round(FRAMES_PER_MONTH / FPS, 1)
SEED = 2027
random.seed(SEED)


def calculate_ms(value_months):
    return int(round(1000 * value_months * SECONDS_PER_MONTH))


def frames_to_ms(i):
    return int(round(1000 * i / FPS))


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

        offense = offense.split("\xa0")
        offense = [of.strip() for of in offense if "..." not in of]
        defence = defence.split("\xa0")
        defence = [of.strip() for of in defence if "..." not in of]
        #
        if finish.lower() == "ongoing":
            finish = "1 January 2025"

        start = parser.parse(start).replace(day=1)
        finish = parser.parse(finish).replace(day=1)
        duration = max(int(round((finish - start) / timedelta(days=30))), 1)

        DURATIONS_TO_FORGET = 3
        MIN_MONTHS_FORGET = 36
        MAX_MONTHS_FORGET = 60

        forget_after_months = min(MAX_MONTHS_FORGET, max(MIN_MONTHS_FORGET, DURATIONS_TO_FORGET * duration))

        forget_after = finish + relativedelta(months=forget_after_months)

        total_duration = forget_after_months + duration

        side_a = offense[0]
        side_b = defence[0]

        for rep in replacements:
            if side_a.strip() == rep:
                side_a = replacements[rep]
            if side_b.strip() == rep:
                side_b = replacements[rep]

        data = {
            "side_a": side_a,
            "side_b": side_b,
            "start": start,
            "finish": finish,
            "num_months": duration,
            "index": index,
            "forget_after": forget_after,
            "total_duration": total_duration,
        }
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

    #"SOVEREIGNT"
    #"SU_A3"
    geometry = country["geometry"]
    c_centroid = centroid(geometry)
    coords = (c_centroid.x.item(), c_centroid.y.item())
    return coords, c_centroid, geometry


def set_image(image, i, start_frame=0, end_frame=0):
    #im = Image.open(image)
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


def handle_ua():
    ukr = gpd.read_file("/home/sunshine/Downloads/ukr_admbnda_sspe_20230201_shp/ukr_admbnda_sspe_20230201_SHP/ukr_admbnda_adm1_sspe_20230201.dbf")

    # crimea_geom = ukr[ukr["ADM1_PCODE"] == "UA01"]["geometry"]
    zpr_geom = ukr[ukr["ADM1_PCODE"] == "UA23"]["geometry"]
    lnr_geom = ukr[ukr["ADM1_PCODE"] == "UA44"]["geometry"]
    khe_geom = ukr[ukr["ADM1_PCODE"] == "UA65"]["geometry"]
    dnr_geom = ukr[ukr["ADM1_PCODE"] == "UA14"]["geometry"]

    return zpr_geom, lnr_geom, khe_geom, dnr_geom


def world_sep_2022(world):
    zpr_geom, lnr_geom, khe_geom, dnr_geom = handle_ua()
    rus_geom = world[world["SU_A3"] == "RUS"]["geometry"]

    ukr_geom = world[world["SU_A3"] == "UKR"]["geometry"]
    merged_polys = unary_union([rus_geom, zpr_geom, lnr_geom, khe_geom, dnr_geom])
    world.loc[world["SU_A3"] == "RUS", ["geometry"]] = merged_polys

    rus_geom = world[world["SU_A3"] == "RUS"]["geometry"]
    ukr_series = gpd.GeoSeries(ukr_geom.iloc[0])
    rus_series = gpd.GeoSeries(rus_geom.iloc[0])

    reduced_ukr_ = ukr_series.difference(rus_series)

    reduced_ukr = MultiPolygon([p for p in reduced_ukr_.item().geoms if p.area > 1])

    add_to_russia = MultiPolygon([p for p in reduced_ukr_.item().geoms if p.area <= 1])
    rus_geom = world[world["SU_A3"] == "RUS"]["geometry"]
    merged_polys_russia = unary_union([add_to_russia, rus_geom.item()])

    world.loc[world["SU_A3"] == "UKR", ["geometry"]] = reduced_ukr
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

    evil_segment = AudioSegment.from_mp3(os.path.join(folder, sounds["evil"]))
    cry_segment = AudioSegment.from_mp3(os.path.join(folder, sounds["cry"]))
    shoot_is = AudioSegment.from_mp3(os.path.join(folder, sounds["shoot_is"]))
    shoot_urban = AudioSegment.from_mp3(os.path.join(folder, sounds["shoot_urban"]))
    shoot_modern = AudioSegment.from_mp3(os.path.join(folder, sounds["shoot_modern"]))
    # silence = AudioSegment.silent(duration=10000)
    # overlay = cry_segment.overlay(evil_segment, position=1000)
    # play(overlay)
    # exit(0)

    im_evil = Image.open(os.path.join(folder, images["evil"]))
    im_cry = Image.open(os.path.join(folder, images["cry"]))
    im_heart = Image.open(os.path.join(folder, images["heart"]))
    im_shoot = Image.open(os.path.join(folder, images["shoot"]))

    world = gpd.read_file("/home/sunshine/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.dbf")

    world_sep22 = world_sep_2022(world)

    def get_world(date):
        if date < datetime(2022, 9, 29):
            return deepcopy(world)
        else:
            # print("Next world")
            return deepcopy(world_sep22)

    fig, ax = plt.subplots(dpi=300, frameon=False)
    # ax.set_aspect('equal')
    # ax.axis("scaled")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    html = "/home/sunshine/Pictures/memes/List of interstate wars since 1945 - Wikipedia.html"
    conflicts = get_conflicts_from_table(html)

    prepare_audio_fn = partial(make_audio_for_conflict, evil_segment=evil_segment, cry_segment=cry_segment,
                               shoot_is=shoot_is, shoot_modern=shoot_modern, shoot_urban=shoot_urban)

    conflicts = [{**conflict, "audio": prepare_audio_fn(conflict)} for conflict in conflicts]

    def animate_conflict(i, num_months, side_a, side_b):

        # ax.set_ylim(-50, 80)
        # ax.set_xlim(-180, 180)
        # world[world["SU_A3"] == side_b].plot(color='orange', ax=ax)
        # evil_color = 'red' if i <= total_months * frames_per_month else "blue"
        # world[world["SU_A3"] == side_a].plot(color=evil_color, ax=ax)

        def plotCountryPatch(axes, country_name, fcolor):
            # plot a country on the provided axes
            nami = world[world["SU_A3"] == country_name]
            namigm = nami.__geo_interface__['features']  # geopandas's geo_interface
            namig0 = {'type': namigm[0]['geometry']['type'], \
                      'coordinates': namigm[0]['geometry']['coordinates']}
            axes.add_patch(PolygonPatch(namig0, fc=fcolor, alpha=0.5, zorder=2))

        plotCountryPatch(ax, side_b, 'red')

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

        #point.plot(ax=ax, marker="o", color="red", markersize=5)
        #point_target.plot(ax=ax, marker="o", color="red", markersize=5)

        is_shooting = i <= num_months * FRAMES_PER_MONTH

        if is_shooting:
            plotCountryPatch(ax, side_a, 'navy')
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
    now = datetime(1993,12,1)  #datetime.now()

    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    max_iters_months = diff_month(now, start_date)
    max_iters = max_iters_months * FRAMES_PER_MONTH

    audio_len_ms = calculate_ms(max_iters_months)
    base_audio = AudioSegment.silent(duration=audio_len_ms)

    with imageio.get_writer('movie.mp4', mode='I', fps=FPS) as writer:
        for i in tqdm(range(max_iters)):
            ax.clear()

            cur_conflicts = get_conflict_by_date(cur_date, conflicts)

            ax.set_ylim(-50, 80)
            ax.set_xlim(-180, 180)
            # ax.set_aspect(2, adjustable="box")

            # world = get_world(cur_date)

            # world.plot(ax=ax)
            #
            if cur_date < datetime(2022, 9, 1):
                world.plot(ax=ax)
            else:
                world_sep22.plot(ax=ax)

            for conflict in cur_conflicts:
                start_index = diff_month(conflict["start"], start_date) * FRAMES_PER_MONTH

                side_a = conflict["side_a"]
                side_b = conflict["side_b"]

                try:
                    params = animate_conflict(i - start_index, conflict["num_months"], side_a, side_b)
                    if params["start_evil"]:
                        base_audio = base_audio.overlay(conflict["audio"], position=frames_to_ms(i))

                except Exception as e:
                    print(e)
                    print(side_a)
                    print(side_b)
                    continue

            # ax.set_title(cur_date.strftime("%Y, %B"))
            ax.text(-170, -45, cur_date.strftime("%Y, %B"), style='italic',
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 0})

            fig.canvas.draw()
            mat = np.array(fig.canvas.renderer._renderer)
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
