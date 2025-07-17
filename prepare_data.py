import pandas as pd
import numpy as np
import pandas as pd

def prepare_data():
    pd.set_option('display.max_rows', None)
    SRU = pd.read_csv('./dataset/SRU.csv', encoding='cp949')
    heatmap_values = pd.read_csv('./dataset/heatmap_values.csv', header=None)
    output_df = pd.read_csv('./dataset/heatmap_center_point.csv', header=None)
    possible_length = int(SRU['speed'] * SRU['time'])
    coverage_drone = SRU['speed'] * SRU['time'] * SRU['coverage']
    total_coverage = 20 * 20
    coverage = total_coverage / coverage_drone
    prob = 1 - np.exp(-coverage)
    return SRU, heatmap_values, output_df, possible_length, coverage, prob