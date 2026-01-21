import random
import utils
import numpy as np


class Greedy:
    def __init__(self, heatmap_values, possible_length, output_df, SRU, init_position, look):
        self.heatmap_values = heatmap_values
        self.possible_length = possible_length
        self.output_df = output_df
        self.SRU = SRU
        self.init_position = init_position
        self.look = look

    def greedy(self, logger, seed, run):
        total_coverage = len(self.heatmap_values) * len(self.heatmap_values)

        random.seed(seed+run)
        greedy_path = []
        # r_idx = random.randint(0, len(self.heatmap_values) - 1)
        # r_inx2 = random.randint(0, len(self.heatmap_values) - 1)

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
        print(x, y)
        greedy_path.append((r_idx, r_inx2))

        detect = 0
        detect += self.heatmap_values.iloc[r_idx, r_inx2]
        print(f"detect = {detect}")
        rows = len(self.heatmap_values)
        cols = len(self.heatmap_values.columns)
        for i in range(self.possible_length - 1):
            actual_coverage = self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time']
            coverage = 1 - np.exp(-(total_coverage / actual_coverage))
            nearby = []
            points = []
            if self.look == '3':
                for i in range(r_idx - 1, r_idx + 2):
                    for j in range(r_inx2 - 1, r_inx2 + 2):
                        if 0 <= i < rows and 0 <= j < cols:
                            nearby.append(self.heatmap_values.iloc[i, j])
                            points.append((i, j))
                # print(f"nearby = {nearby}")
                # print(f"points = {points}")
                idx_max_value = nearby.index(max(nearby))
                greedy_path.append(points[idx_max_value])
                detect += self.heatmap_values.iloc[points[idx_max_value]]
                self.heatmap_values.iloc[points[idx_max_value]] = int(
                    self.heatmap_values.iloc[points[idx_max_value]] * (1 - coverage))

                r_idx = points[idx_max_value][0]
                r_inx2 = points[idx_max_value][1]


            elif self.look == '5':
                for dx in [-1, 0, 1]:
                    # print(f"r_idx = {r_idx}, r_idx2 = {r_inx2}")
                    for dy in [-1, 0, 1]:
                        i1 = r_idx + dx
                        j1 = r_inx2 + dy
                        i2 = r_idx + 2 * dx
                        j2 = r_inx2 + 2 * dy
                        # print(f"i1 = {i1}, j1 = {j1}, i2 = {i2}, j2 = {j2}")
                        val1 = 0
                        val2 = 0

                        if 0 <= i1 < rows and 0 <= j1 < cols:
                            val1 = self.heatmap_values.iloc[i1, j1]
                            # if val1 == 0:
                            #     val1 += 1
                            # print(f"val1 = {val1}")
                        if 0 <= i2 < rows and 0 <= j2 < cols:
                            val2 = self.heatmap_values.iloc[i2, j2]
                            # if val2 == 0:
                            #     val2 += 1
                            # print(f"val2 = {val2}")
                        weight = 0.8
                        score = weight * val1 + (1-weight) * val2

                        nearby.append(score)
                        points.append((i1, j1))
                # print(f"nearby = {nearby}")
                # print(f"points = {points}")
                idx_max_value = nearby.index(max(nearby))
                greedy_path.append(points[idx_max_value])
                detect += self.heatmap_values.iloc[points[idx_max_value]]
                # print(f"coordination = {points[idx_max_value]}")
                # print(f"before = {self.heatmap_values.iloc[points[idx_max_value]]}")
                self.heatmap_values.iloc[points[idx_max_value]] = int(self.heatmap_values.iloc[points[idx_max_value]] * (1 - coverage))
                # print(f"after = {self.heatmap_values.iloc[points[idx_max_value]]}")
                r_idx = points[idx_max_value][0]
                r_inx2 = points[idx_max_value][1]

        utils.print_and_log(logger, f"Greedy values: {greedy_path}")
        final_path = []
        for i in range(len(greedy_path)):
            final_path.append(self.output_df.iloc[greedy_path[i]])

        utils.print_and_log(logger, f"final path: {final_path}")
        utils.print_and_log(logger, f"detected values: {detect}")
        return final_path, detect
    # [수정] run=0 기본값 추가, custom_start 추가
    def greedy_rl(self, logger, seed, run=0, custom_start=None):
        total_coverage = len(self.heatmap_values) * len(self.heatmap_values)

        random.seed(seed + run)
        greedy_path = []

        # [수정] 랜덤 시작점이 있으면 사용, 없으면 설정값 사용
        if custom_start is not None:
            x, y = custom_start
        else:
            if self.init_position == "leftup": x, y = 2, 3
            elif self.init_position == "leftdown": x, y = 19, 0
            elif self.init_position == "rightup": x, y = 3, 15
            elif self.init_position == "rightdown": x, y = 18, 13
            elif self.init_position == "center": x, y = 10, 10

        r_idx = x
        r_inx2 = y
        greedy_path.append((r_idx, r_inx2))

        detect = 0
        # [수정] 스칼라 값 접근 안전하게 변경
        detect += self.heatmap_values.values[r_idx, r_inx2]
        
        rows = len(self.heatmap_values)
        cols = len(self.heatmap_values.columns)
        
        for i in range(self.possible_length - 1):
            actual_coverage = self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time']
            coverage = 1 - np.exp(-(total_coverage / actual_coverage))
            nearby = []
            points = []
            
            if self.look == '3':
                for i in range(r_idx - 1, r_idx + 2):
                    for j in range(r_inx2 - 1, r_inx2 + 2):
                        if 0 <= i < rows and 0 <= j < cols:
                            nearby.append(self.heatmap_values.iloc[i, j])
                            points.append((i, j))
                
                if not nearby: break
                
                idx_max_value = nearby.index(max(nearby))
                greedy_path.append(points[idx_max_value])
                
                # [수정] 튜플 언패킹으로 에러 방지
                r_next, c_next = points[idx_max_value]
                detect += self.heatmap_values.iloc[r_next, c_next]
                
                self.heatmap_values.iloc[r_next, c_next] = int(
                    self.heatmap_values.iloc[r_next, c_next] * (1 - coverage)
                )

                r_idx = r_next
                r_inx2 = c_next


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
                            val1 = self.heatmap_values.iloc[i1, j1]
                        if 0 <= i2 < rows and 0 <= j2 < cols:
                            val2 = self.heatmap_values.iloc[i2, j2]
                        
                        weight = 0.8
                        score = weight * val1 + (1-weight) * val2

                        nearby.append(score)
                        points.append((i1, j1))
                
                if not nearby: break

                idx_max_value = nearby.index(max(nearby))
                greedy_path.append(points[idx_max_value])
                
                # [수정] 튜플 언패킹으로 에러 방지
                r_next, c_next = points[idx_max_value]
                detect += self.heatmap_values.iloc[r_next, c_next]
                
                self.heatmap_values.iloc[r_next, c_next] = int(
                    self.heatmap_values.iloc[r_next, c_next] * (1 - coverage)
                )
                
                r_idx = r_next
                r_inx2 = c_next

        if logger:
            utils.print_and_log(logger, f"Greedy values: {greedy_path}")
        
        final_path = []
        for i in range(len(greedy_path)):
            final_path.append(self.output_df.iloc[greedy_path[i]])

        if logger:
            utils.print_and_log(logger, f"final path: {final_path}")
            utils.print_and_log(logger, f"detected values: {detect}")
        
        # [중요 수정] RL 정책 업데이트를 위해 '좌표 리스트(greedy_path)'를 반환해야 함!
        return greedy_path, detect