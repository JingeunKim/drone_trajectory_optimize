import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def setup_logger(arg):
    logger = logging.getLogger()
    log_path = './logs/{:%Y%m%d}_{}_{}_{}.log'.format(datetime.datetime.now(), arg.mode, arg.position, arg.look)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def print_and_log(logger, msg):
    # global logger
    print(msg)
    logger.info(msg)

def draw(final_path, args):
    print("draw phase")
    heatmap = pd.read_csv("./dataset/heatmap_values.csv", header=None)
    df = pd.read_csv("./dataset/heatmap_center_point.csv", header=None)
    print(heatmap)
    print(len(heatmap))
    height, width = heatmap.shape

    lons = []
    lats = []


    for location in final_path:
        lon, lat = map(float, location.split('/'))
        lons.append(lon)
        lats.append(lat)

    lon_list = []
    lat_list = []

    for target in final_path:
        for i in range(df.shape[0]):  # 행
            for j in range(df.shape[1]):  # 열
                if df.iat[i, j] == target:
                    lat_list.append(i)  # 행 → y (위도)
                    lon_list.append(j)  # 열 → x (경도)
                    break
            else:
                continue
            break

    # lat_list = [19 - lat for lat in lat_list]

    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    x_indices = []
    y_indices = []
    for lon, lat in zip(lon_list, lat_list):
        x_indices.append(lon + 0.5)
        y_indices.append(lat + 0.5)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        heatmap,
        cmap='YlOrRd',  # 'viridis' 말고 밝은 걸로 바꾸는 것도 고려
        cbar=True,
        linewidths=0.5,  # 셀 테두리 두께
        linecolor='gray'  # 셀 테두리 색
    )

    plt.plot(x_indices, y_indices, color='red', marker='o', linewidth=2, label='Path')
    position_counts = {}

    for i, (x, y) in enumerate(zip(x_indices, y_indices)):
        key = (x, y)
        if key in position_counts:
            position_counts[key] += 1
        else:
            position_counts[key] = 0

        offset = position_counts[key] * 0.3  # 중복되면 위로 0.3씩 올림
        plt.text(x, y - offset, str(i + 1), color='white', fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='red', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.6))

    xtick_positions = np.linspace(0, width - 1, num=6)
    ytick_positions = np.linspace(0, height - 1, num=6)

    xtick_labels = [f"{lon_min + (lon_max - lon_min) * (x / (width - 1)):.4f}" for x in xtick_positions]
    ytick_labels = [f"{lat_min + (lat_max - lat_min) * (y / (height - 1)):.4f}" for y in ytick_positions]

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)

    # ax.invert_yaxis()  # 이걸 꼭 추가!

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()

    save_folder = './Figure/'+ args.mode + "_" + args.position
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, 'result_run_' + str(args.run) + "_" + str(args.look) + '.png')
    plt.savefig(save_path)
    # plt.show()

def latlon_to_index(lon, lat, lon_min, lon_max, lat_min, lat_max, width, height):
    x_idx = int((lon - lon_min) / (lon_max - lon_min) * (width - 1))
    y_idx = int((lat - lat_min) / (lat_max - lat_min) * (height - 1))
    return x_idx, y_idx