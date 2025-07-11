import random
import numpy as np
import ga_for_hybrid
import copy


class hybrid:
    def __init__(self, heatmap_values, possible_length, output_df, SRU, args):
        self.heatmap_values = heatmap_values
        self.possible_length = possible_length
        self.output_df = output_df
        self.SRU = SRU
        self.generations = args.generation
        self.pop_size = args.pop_size
        self.init_position = args.position
        self.muation_rate = args.mutation_rate
        self.local_search = args.mode
        self.look = args.look
        self.init_step = 35
        self.step = 40
        self.args = args

    def greedy(self, r_idx, r_inx2, total_coverage, greedy_path, length, heatmap_values):
        detect = 0
        detect += heatmap_values.iloc[r_idx, r_inx2]
        rows = len(heatmap_values)
        cols = len(heatmap_values.columns)
        # print(f"in greedy length = {length}")
        for i in range(length - 1):
            actual_coverage = self.SRU['speed'] * self.SRU['coverage'] * i * self.SRU['time']
            coverage = 1 - np.exp(-(total_coverage / actual_coverage))
            nearby = []
            points = []
            if self.look == '3':
                for i in range(r_idx - 1, r_idx + 2):
                    for j in range(r_inx2 - 1, r_inx2 + 2):
                        if 0 <= i < rows and 0 <= j < cols:
                            nearby.append(heatmap_values.iloc[i, j])
                            points.append((i, j))
                idx_max_value = nearby.index(max(nearby))
                greedy_path.append(points[idx_max_value])
                detect += heatmap_values.iloc[points[idx_max_value]]
                heatmap_values.iloc[points[idx_max_value]] = int(
                    heatmap_values.iloc[points[idx_max_value]] * (1 - coverage))

                r_idx = points[idx_max_value][0]
                r_inx2 = points[idx_max_value][1]


            elif self.look == '5':
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        i1 = r_idx + dx
                        j1 = r_inx2 + dy
                        i2 = r_idx + 2 * dx
                        j2 = r_inx2 + 2 * dy
                        val1 = 0
                        val2 = 0

                        if 0 <= i1 < rows and 0 <= j1 < cols:
                            val1 = heatmap_values.iloc[i1, j1]
                            if val1 == 0:
                                val1 += 1
                        if 0 <= i2 < rows and 0 <= j2 < cols:
                            val2 = heatmap_values.iloc[i2, j2]
                            if val2 == 0:
                                val2 += 1
                        score = 0.8 * val1 + 0.2 * val2

                        nearby.append(score)
                        points.append((i1, j1))
                idx_max_value = nearby.index(max(nearby))
                greedy_path.append(points[idx_max_value])
                detect += heatmap_values.iloc[points[idx_max_value]]
                heatmap_values.iloc[points[idx_max_value]] = int(
                    heatmap_values.iloc[points[idx_max_value]] * (1 - coverage))
                r_idx = points[idx_max_value][0]
                r_inx2 = points[idx_max_value][1]
        return greedy_path, heatmap_values

    def final_eval(self, individual, total_coverage):
        detect_point = 0
        print("final eval")
        print(self.heatmap_values)
        heatmap_values_copy = copy.deepcopy(self.heatmap_values)
        for i in range(self.possible_length - 1):
            actual_coverage = self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time']
            coverage = 1 - np.exp(-(total_coverage / actual_coverage))
            detect_point += heatmap_values_copy.iloc[individual[i]]
            heatmap_values_copy.iloc[individual[i]] = int(heatmap_values_copy.iloc[individual[i]] * (1 - coverage))

        return detect_point

    def run_greedy_and_ga(self, r_idx, r_inx2, total_coverage, greedy_path, step, heatmap, logger, seed, init_step = 15):
        g_path, heatmap = self.greedy(r_idx, r_inx2, total_coverage, greedy_path, init_step, heatmap)
        g_x, g_y = g_path[-1]
        # print("run_greedy_and_ga")
        # print(f"g_path: {g_path}, g_x: {g_x}, g_y: {g_y}")
        ga = ga_for_hybrid.GA(heatmap, step, self.output_df, self.SRU, self.args, g_x, g_y)
        ga_path, heatmap = ga.evolve(logger, seed)

        g_x2, g_y2 = ga_path[-1]
        return ga_path, heatmap, g_path, g_x2, g_y2

    def find_path(self, logger, seed):
        total_coverage = len(self.heatmap_values) * len(self.heatmap_values)

        random.seed(seed)
        genetic_path = []
        greedy_path = []

        if self.init_position == "leftup":
            x = 2
            y = 3
        elif self.init_position == "leftdown":
            x = 19
            y = 0
        elif self.init_position == "rightup":
            x = 3
            y = 15
        elif self.init_position == "rightdown":
            x = 18
            y = 13
        elif self.init_position == "center":
            x = 10
            y = 10

        r_idx = x
        r_inx2 = y
        greedy_path.append((r_idx, r_inx2))
        # print(f"init = {r_idx}, {r_inx2}")
        # 첫 번째 GA 실행
        g_path, heatmap = self.greedy(r_idx, r_inx2, total_coverage, greedy_path, self.init_step, self.heatmap_values)
        # print(f"g_path123123123: {len(g_path)}")
        g_x, g_y = g_path[-1]
        # print(f"g_path123123123: {len(g_path)}")
        ga = ga_for_hybrid.GA(heatmap, self.step, self.output_df, self.SRU, self.args, g_x, g_y)
        # print(f"g_path123123123: {len(g_path)}")
        ga_path, heatmap = ga.evolve(logger, seed)
        # print(f"g_path123123123: {len(g_path)}")
        # g_x2, g_y2 = ga_path[-1]
        # greedy_path2 = [ga_path[-1]]
        # # print(f"greedy_path2: {greedy_path2}")
        # # print(f"g_path123123123: {len(g_path)}")
        # g_path2, heatmap = self.greedy(g_x2, g_y2, total_coverage, greedy_path2, self.step, heatmap)
        # # print(f"g_path123123123: {len(g_path)}")
        # g_x3, g_y3 = g_path2[-1]
        # # print(f"g_path123123123: {len(g_path)}")
        # ga2 = ga_for_hybrid.GA(heatmap, self.step, self.output_df, self.SRU, self.args, g_x3, g_y3)
        # # print(f"g_path123123123: {len(g_path)}")
        # ga_path2, heatmap = ga2.evolve(logger, seed)

        # print(f"123123123123123: {len(g_path)}")
        # print(len(g_path))
        # print(len(ga_path))
        # print(len(g_path2))
        # print(len(ga_path2))

        # print("-"*40)
        # print(g)
        # print(ga_path)
        # print(g2)
        # print(ga_path2)

        all_path = g_path + ga_path #+ g_path2 + ga_path2
        print("=" * 40)
        print(len(g_path))
        print(len(ga_path))
        # print(len(g_path2))
        # print(len(ga_path2))
        print(all_path)
        print(len(all_path))
        num_detect = self.final_eval(all_path, total_coverage)
        print(num_detect)

        final_path = []
        for i in range(len(all_path)):
            final_path.append(self.output_df.iloc[all_path[i]])

        return final_path, num_detect