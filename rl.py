import random
import matplotlib.pyplot as plt
import seaborn as sns
from greedy import Greedy
import utils
import numpy as np


class rl:
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
        self.mode = args.RLmode
        self.inherit = args.inherit
        self.crossover_mode = args.crossover_mode
        self.strength = args.strength
        self.n_episode = 10000
        self.length_episode = 75
        self.rho = 0.3  # 학습률
        self.lamda = 0.99  # 할인율
        self.actions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

    def init_p(self):
        x = 0
        y = 0
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
        return x, y

    def pos_to_idx(self, x, y, W):
        return x * W + y

    def is_valid(self, x, y, H, W):
        return 0 <= x < H and 0 <= y < W

    def _select_parents_random(self, population):
        return random.sample(population, 2)

    def _crossover(self, parent1, parent2, H, W):
        p1_policy, p1_critic = parent1
        p2_policy, p2_critic = parent2

        # print(f"p1_critic: {p1_critic} len = {len(p1_critic)}")
        # print(f"p1_critic[0]: {p1_critic[0]}")
        row_pt = random.randint(0, H - 1)
        col_pt = random.randint(0, H - 1)
        p1_pol_grid = p1_policy.reshape(H, W, 8)
        p2_pol_grid = p2_policy.reshape(H, W, 8)

        p1_val_grid = p1_critic.reshape(H, W)
        p2_val_grid = p2_critic.reshape(H, W)
        # print(f"row_pt: {row_pt}")
        # print("p1_pol_grid")
        # print(p1_pol_grid)
        # print("p1_pol_grid 쪼개기")
        # print(p1_pol_grid[col_pt:, row_pt:])
        child_pol_grid = np.zeros((H, W, 8))
        child_val_grid = np.zeros((H, W))
        # 1사분면 (우상) -> Parent 1
        child_pol_grid[:row_pt, col_pt:] = p1_pol_grid[:row_pt, col_pt:]
        child_val_grid[:row_pt, col_pt:] = p1_val_grid[:row_pt, col_pt:]

        # 3사분면 (좌하) -> Parent 1
        child_pol_grid[row_pt:, :col_pt] = p1_pol_grid[row_pt:, :col_pt]
        child_val_grid[row_pt:, :col_pt] = p1_val_grid[row_pt:, :col_pt]

        # 2사분면 (좌상) -> Parent 2
        child_pol_grid[:row_pt, :col_pt] = p2_pol_grid[:row_pt, :col_pt]
        child_val_grid[:row_pt, :col_pt] = p2_val_grid[:row_pt, :col_pt]

        # 4사분면 (우하) -> Parent 2
        child_pol_grid[row_pt:, col_pt:] = p2_pol_grid[row_pt:, col_pt:]
        child_val_grid[row_pt:, col_pt:] = p2_val_grid[row_pt:, col_pt:]

        # 5. 다시 원래 모양(Flatten)으로 되돌리기
        child_policy = child_pol_grid.reshape(H * W, 8)
        child_critic = child_val_grid.reshape(H * W)
        return (child_policy, child_critic)


    def _crossover_row(self, parent1, parent2, H, W):
            p1_policy, p1_critic = parent1
            p2_policy, p2_critic = parent2

            p1_pol_grid = p1_policy.reshape(H, W, 8)
            p2_pol_grid = p2_policy.reshape(H, W, 8)
            p1_val_grid = p1_critic.reshape(H, W)
            p2_val_grid = p2_critic.reshape(H, W)

            child_pol_grid = np.zeros((H, W, 8))
            child_val_grid = np.zeros((H, W))

            cut_row = random.randint(1, H - 1)

            child_pol_grid[:cut_row, :] = p1_pol_grid[:cut_row, :]
            child_val_grid[:cut_row, :] = p1_val_grid[:cut_row, :]

            child_pol_grid[cut_row:, :] = p2_pol_grid[cut_row:, :]
            child_val_grid[cut_row:, :] = p2_val_grid[cut_row:, :]
            child_policy = child_pol_grid.reshape(H * W, 8)
            child_critic = child_val_grid.reshape(H * W)
            
            return (child_policy, child_critic)
    
    def _crossover_col(self, parent1, parent2, H, W):
        p1_policy, p1_critic = parent1
        p2_policy, p2_critic = parent2

        p1_pol_grid = p1_policy.reshape(H, W, 8)
        p2_pol_grid = p2_policy.reshape(H, W, 8)
        p1_val_grid = p1_critic.reshape(H, W)
        p2_val_grid = p2_critic.reshape(H, W)

        child_pol_grid = np.zeros((H, W, 8))
        child_val_grid = np.zeros((H, W))

        cut_col = random.randint(1, W - 1)

        child_pol_grid[:, :cut_col] = p1_pol_grid[:, :cut_col]
        child_val_grid[:, :cut_col] = p1_val_grid[:, :cut_col]

        child_pol_grid[:, cut_col:] = p2_pol_grid[:, cut_col:]
        child_val_grid[:, cut_col:] = p2_val_grid[:, cut_col:]

        child_policy = child_pol_grid.reshape(H * W, 8)
        child_critic = child_val_grid.reshape(H * W)
        
        return (child_policy, child_critic)
        ''
    def _mutate(self, individual, mutation_rate, H, W):
        policy, critic = individual
        for s in range(H * W):
            if random.random() < mutation_rate:
                noise = np.random.normal(0, 0.1, size=8)
                policy[s] += noise
                policy[s] = np.maximum(policy[s], 1e-8)
                policy[s] /= np.sum(policy[s])

        for s in range(H * W):
            if random.random() < mutation_rate:
                noise = np.random.normal(0, 0.1)
                critic[s] += noise

    def find_path(self, logger, seed, run):
        random.seed(seed+run)
        np.random.seed(seed+run)
        if self.mode == "q":
            total_coverage = len(self.heatmap_values) * len(self.heatmap_values)
            actual_coverage = self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time']
            coverage = 1 - np.exp(-(total_coverage / actual_coverage))

            final_path = []
            best_reward = -1

            H, W = self.heatmap_values.shape
            Q = np.zeros((H * W, 8))
            print(f"Q.shape: {Q.shape}")
            actions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
            # reward_list = []
            epsilon = 0.9
            for ep in range(self.n_episode):
                # print(f"{ep} episode")
                x, y = self.init_p()
                # print(f"initial position: {x}, y: {y}")
                visited = np.zeros((H, W), dtype=bool)
                path = [(x, y)]
                state = self.pos_to_idx(x, y, W)
                # print(f"init state: {state}")
                total_reward = 0
                heatmap_values_copy = self.heatmap_values.copy(deep=True)

                if ep % 10 == 0 and ep != 0:
                    epsilon *= 0.9
                    epsilon = max(epsilon, 0.1)

                for t in range(self.length_episode):
                    if random.random() < epsilon:
                        a = random.randint(0, 7)
                        # print(f"11 a = {a}")
                    else:
                        # print(f"Q[state] = {Q[state]}")
                        # a = np.argmax(Q[state])
                        if np.all(Q[state] == 0):
                            a = random.choice(range(len(Q[state])))
                        else:
                            a = np.argmax(Q[state])
                        # print(f"22 a = {a}")

                    dx, dy = actions[a]
                    # print(f"action[a] = {actions[a]}")
                    # print(f"="*50)
                    # print(f"dx = {dx}, dy = {dy}")
                    nx, ny = x + dx, y + dy
                    if not self.is_valid(nx, ny, H, W):
                        nx, ny = x, y

                    if (nx, ny) in visited:
                        candidates = []
                        for alt_action in range(8):
                            adx, ady = self.actions[alt_action]
                            ax, ay = x + adx, y + ady
                            if self.is_valid(ax, ay, H, W) and (ax, ay) not in visited:
                                candidates.append((alt_action, ax, ay))

                        # if candidates:
                        #     alt_action, ax, ay = random.choice(candidates)
                        #     nx, ny = ax, ay
                        #     a = alt_action
                        if candidates:
                            # 후보지들 중 Heatmap 값(보상)이 가장 높은 곳을 선택 (탐욕적 회피)
                            # 또는 policy[state] 확률이 가장 높은 곳을 선택해도 됨
                            best_cand = None
                            max_cand_val = -float('inf')

                            for cand in candidates:
                                c_act, c_x, c_y = cand
                                # 지금 당장 먹을 수 있는 점수를 확인
                                val = heatmap_values_copy.iloc[c_x, c_y] 
                                if val > max_cand_val:
                                    max_cand_val = val
                                    best_cand = cand
                            
                            # 만약 모든 후보지의 가치가 같다면(0이라면) 그냥 아무거나
                            if best_cand is None:
                                best_cand = random.choice(candidates)

                            alt_action, ax, ay = best_cand
                            nx, ny = ax, ay
                            action = alt_action


                    # print(f"nx = {nx}, ny = {ny}")
                    # print(f"next = {nx, ny}")

                    if not self.is_valid(nx, ny, H, W):
                        nx, ny = x, y
                    # print(f"x = {x}, y = {y}")
                    reward = heatmap_values_copy.iloc[nx, ny]
                    heatmap_values_copy.iloc[nx, ny] = int(reward * (1 - coverage))
                    # print(f"reward = = {reward}")
                    # print(f"bb = = {heatmap_values_copy.iloc[nx, ny]}")
                    # print(f"aa = = {heatmap_values_copy.iloc[nx, ny]}")
                    # if visited[nx, ny]:
                    #     reward = 0
                    # visited[nx, ny] = True

                    next_state = self.pos_to_idx(nx, ny, W)
                    # print(f"reward = {reward}, Q[next_state] = {Q[next_state]}, Q[state, a] = {Q[state, a]}")
                    Q[state, a] += self.rho * (reward + self.lamda * np.max(Q[next_state]) - Q[state, a])
                    # print(f"after Q[next_state] = {Q[next_state]}")
                    # print(f"after Q[state, a] = {Q[state, a]}")
                    # print(f"Q[state, a] = {Q[state, a]}")

                    x, y = nx, ny
                    # print(f"new x = {x}, new y = {y}")
                    state = next_state
                    path.append((x, y))
                    total_reward += reward

                if total_reward > best_reward:
                    best_reward = total_reward
                    final_path = path
                    print(f"[EP {ep}] Best Reward: {best_reward}")
                if ep % 100 == 0:
                    print(f"[EP {ep}] Reward: {total_reward}")
                # reward_list.append(total_reward)
                # avg_reward = np.mean(reward_list)
                # max_reward = np.max(reward_list)
                # print(f"Average reward over all episodes: {avg_reward:.4f}")
                # print(f"Maximum reward achieved: {max_reward:.4f}")

            print(f"Q table")
            # self.visualize_q_table(Q, H, W)
            print(Q)
            print(Q.shape)
            print(f"final_path = {final_path}")
            final_path_po = []
            for i in range(len(final_path)):
                final_path_po.append(self.output_df.iloc[final_path[i]])

            return final_path_po, best_reward

        elif self.mode == 'ac':
            H, W = self.heatmap_values.shape
            policy = np.full((H * W, 8), 1 / 8)  # Actor: 각 상태에서의 행동 확률
            print(f"policy = {policy}")
            V = np.zeros(H * W)  # Critic: 상태 가치 함수
            print(f"V = {V}")

            alpha = 0.1
            beta = 0.01
            gamma = 0.99
            epsilon = 0.9

            best_reward = -1
            final_path = []

            for episode in range(self.n_episode):
                x, y = self.init_p()
                state = self.pos_to_idx(x, y, W)
                total_reward = 0
                path = [(x, y)]
                heatmap_values_copy = self.heatmap_values.copy(deep=True)
                visited = set()
                visited.add((x, y))

                if episode % 100 == 0 and episode != 0:
                    epsilon *= 0.9
                    epsilon = max(epsilon, 0.1)

                for t in range(self.length_episode):
                    if random.random() < epsilon:
                        action = random.randint(0, 7)
                    else:
                        if np.all(policy[state] == 0):
                            # 만약 정책이 비어있다면 랜덤 이동 (에러 방지용)
                            action = random.choice(range(len(policy[state])))
                        else:
                            action = np.random.choice(len(policy[state]), p=policy[state])

                    dx, dy = self.actions[action]
                    nx, ny = x + dx, y + dy
                    if not self.is_valid(nx, ny, H, W):
                        nx, ny = x, y

                    if (nx, ny) in visited:
                        candidates = []
                        for alt_action in range(8):
                            adx, ady = self.actions[alt_action]
                            ax, ay = x + adx, y + ady
                            if self.is_valid(ax, ay, H, W) and (ax, ay) not in visited:
                                candidates.append((alt_action, ax, ay))

                        # if candidates:
                        #     alt_action, ax, ay = random.choice(candidates)
                        #     nx, ny = ax, ay
                        #     action = alt_action
                        if candidates:
                            # 후보지들 중 Heatmap 값(보상)이 가장 높은 곳을 선택 (탐욕적 회피)
                            # 또는 policy[state] 확률이 가장 높은 곳을 선택해도 됨
                            best_cand = None
                            max_cand_val = -float('inf')

                            for cand in candidates:
                                c_act, c_x, c_y = cand
                                # 지금 당장 먹을 수 있는 점수를 확인
                                val = heatmap_values_copy.iloc[c_x, c_y] 
                                if val > max_cand_val:
                                    max_cand_val = val
                                    best_cand = cand
                            
                            # 만약 모든 후보지의 가치가 같다면(0이라면) 그냥 아무거나
                            if best_cand is None:
                                best_cand = random.choice(candidates)

                            alt_action, ax, ay = best_cand
                            nx, ny = ax, ay
                            action = alt_action

                    reward = heatmap_values_copy.iloc[nx, ny]
                    coverage = 1 - np.exp(-(H * W / (self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time'])))
                    heatmap_values_copy.iloc[nx, ny] = int(reward * (1 - coverage))

                    next_state = self.pos_to_idx(nx, ny, W)

                    # Critic update (TD error)
                    td_error = reward + gamma * V[next_state] - V[state]
                    V[state] += alpha * td_error

                    # Actor update (policy gradient)
                    policy[state, action] += beta * td_error
                    policy[state] = np.maximum(policy[state], 1e-8)  # prevent negative
                    policy[state] /= np.sum(policy[state])  # normalize

                    x, y = nx, ny
                    state = next_state
                    path.append((x, y))
                    visited.add((x, y))
                    total_reward += reward

                utils.print_and_log(logger, f"fitness = {best_reward:.2f}")
                if total_reward > best_reward:
                    best_reward = total_reward
                    final_path = path
                    print(f"[EP {episode}] Best Reward: {best_reward}")
                if episode % 100 == 0:
                    print(f"[EP {episode}] Reward: {total_reward}")

            print(f"final_path = {final_path}")
            final_path_po = []
            for i in range(len(final_path)):
                final_path_po.append(self.output_df.iloc[final_path[i]])

            return final_path_po, best_reward

        elif self.mode == 'ac_ea':
            H, W = self.heatmap_values.shape
            population_size = self.pop_size
            n_generations = self.generations
            mutation_rate = self.muation_rate

            alpha = 0.1
            beta = 0.01
            gamma = 0.99

            population = []

            for _ in range(population_size):
                policy = np.full((H * W, 8), 1.0 / 8.0)
                V = np.zeros(H * W)
                population.append((policy, V))
            print(f"population = {population[0]}")
            best_reward_overall = -float('inf')
            final_path_overall = []

            print(f"--- 시작: Actor-Critic + Evolutionary Algorithm (AC-EA) ---")
            print(f"세대 수: {n_generations}, 개체 수: {population_size}, 변이율: {mutation_rate}")

            for gen in range(n_generations):
                fitness_scores = []

                all_paths = []
                for i in range(population_size):
                    policy_, V_ = population[i]
                    if self.inherit == "darwin": 
                        policy = policy_.copy()
                        V = V_.copy()
                    else:
                        policy = policy_    
                        V = V_

                    x, y = self.init_p()
                    state = self.pos_to_idx(x, y, W)
                    total_reward = 0
                    path = [(x, y)]
                    heatmap_values_copy = self.heatmap_values.copy(deep=True)
                    visited = set()
                    visited.add((x, y))

                    for t in range(self.length_episode):
                        action = np.random.choice(len(policy[state]), p=policy[state])
                        # if np.all(policy[state] == 0):
                        #     action = random.choice(range(len(policy[state])))
                        # else:
                        #     if np.random.random() < 0.2:
                        #         action = np.random.choice(len(policy[state]), p=policy[state])
                        #     else:
                        #         action = np.argmax(policy[state])

                        dx, dy = self.actions[action]
                        nx, ny = x + dx, y + dy

                        if not self.is_valid(nx, ny, H, W):
                            nx, ny = x, y

                        if (nx, ny) in visited:
                            candidates = []

                            for alt_action in range(8):
                                adx, ady = self.actions[alt_action]
                                ax, ay = x + adx, y + ady
                                if self.is_valid(ax, ay, H, W) and (ax, ay) not in visited:
                                    candidates.append((alt_action, ax, ay))

                            # if candidates:
                            #     alt_action, ax, ay = random.choice(candidates)
                            #     nx, ny = ax, ay
                            #     action = alt_action
                            if candidates:
                            # 후보지들 중 Heatmap 값(보상)이 가장 높은 곳을 선택 (탐욕적 회피)
                            # 또는 policy[state] 확률이 가장 높은 곳을 선택해도 됨
                                best_cand = None
                                max_cand_val = -float('inf')

                                for cand in candidates:
                                    c_act, c_x, c_y = cand
                                    # 지금 당장 먹을 수 있는 점수를 확인
                                    val = heatmap_values_copy.iloc[c_x, c_y] 
                                    if val > max_cand_val:
                                        max_cand_val = val
                                        best_cand = cand
                                
                                # 만약 모든 후보지의 가치가 같다면(0이라면) 그냥 아무거나
                                if best_cand is None:
                                    best_cand = random.choice(candidates)

                                alt_action, ax, ay = best_cand
                                nx, ny = ax, ay
                                action = alt_action

                        reward = heatmap_values_copy.iloc[nx, ny]
                        coverage = 1 - np.exp(-(H * W / (self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time'])))
                        heatmap_values_copy.iloc[nx, ny] = int(reward * (1 - coverage))

                        next_state = self.pos_to_idx(nx, ny, W)
                        td_error = reward + gamma * V[next_state] - V[state]

                        V[state] += alpha * td_error
                        policy[state, action] += beta * td_error
                        policy[state] = np.maximum(policy[state], 1e-8)
                        policy[state] /= np.sum(policy[state])

                        x, y = nx, ny
                        state = next_state
                        path.append((x, y))
                        visited.add((x, y))
                        total_reward += reward

                    fitness_scores.append(total_reward)
                    all_paths.append(path)
                    if total_reward > best_reward_overall:
                        best_reward_overall = total_reward
                        final_path_overall = path

                avg_fitness = np.mean(fitness_scores)
                max_fitness = np.max(fitness_scores)
                print(
                    f"[세대 {gen + 1}/{n_generations}] 평균 적합도: {avg_fitness:.2f}, 최고 적합도: {max_fitness:.2f}, (전체 최고: {best_reward_overall:.2f})")
                utils.print_and_log(logger, f"fitness = {best_reward_overall:.2f}")

                new_population = []
                num_elites = int(self.pop_size * 0.2)
                elite_indices = np.argsort(fitness_scores)[-num_elites:]

                for idx in elite_indices:
                    elite_policy = population[idx][0].copy()
                    elite_critic = population[idx][1].copy()
                    new_population.append((elite_policy, elite_critic))

                parent_indices = list(range(self.pop_size))
                random.shuffle(parent_indices)
                idx_pointer = 0
                ############랜덤샐랙션#################
                ####################################
                # while len(new_population) < population_size:
                #     if idx_pointer >= population_size - 1:
                #         random.shuffle(parent_indices)
                #         idx_pointer = 0
                #
                #     p1_idx = parent_indices[idx_pointer]
                #     p2_idx = parent_indices[idx_pointer + 1]
                #     parent1 = population[p1_idx]
                #     parent2 = population[p2_idx]


                ############토너먼트샐랙션#################
                ####################################
                while len(new_population) < population_size:
                    # 부모 1: 랜덤 5명 뽑아서 그 중 1등 선택 (경쟁)
                    cands1 = random.sample(range(population_size), 5)
                    p1_idx = max(cands1, key=lambda i: fitness_scores[i])
                    # print(f"p1 = {p1_idx}")
                    # 부모 2: 랜덤 5명 뽑아서 그 중 1등 선택
                    cands2 = random.sample(range(population_size), 5)
                    p2_idx = max(cands2, key=lambda i: fitness_scores[i])
                    # print(f"p2 = {p2_idx}")
                    # (선택) 엄마 아빠 같으면 다시 뽑기
                    if p1_idx == p2_idx:
                        cands2 = random.sample(range(population_size), 5)
                        p2_idx = max(cands2, key=lambda i: fitness_scores[i])
                    # print(f"p1 = {p1_idx}, p2 = {p2_idx}")

                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]
                    if self.crossover_mode == "quarter":
                        child1 = self._crossover(parent1, parent2, H, W)
                    elif self.crossover_mode == "row":
                        child1 = self._crossover_row(parent1, parent2, H, W)
                    elif self.crossover_mode == "col":
                        child1 = self._crossover_col(parent1, parent2, H, W)
                    self._mutate(child1, mutation_rate, H, W)
                    if len(new_population) < population_size:
                        new_population.append(child1)
                    idx_pointer += 2
                # center 8389
                population = new_population
            print(f"--- 진화 완료. 최종 최고 보상: {best_reward_overall} ---")
            print(f"final_path = {final_path_overall}")

            final_path_po = []

            for i in range(len(final_path_overall)):
                final_path_po.append(self.output_df.iloc[final_path_overall[i]])

            return final_path_po, best_reward_overall
        
        elif self.mode == 'ac_ma':
            H, W = self.heatmap_values.shape
            population_size = self.pop_size
            n_generations = self.generations
            mutation_rate = self.muation_rate

            alpha = 0.1
            beta = 0.01
            gamma = 0.99

            population = []

            for _ in range(population_size):
                policy = np.full((H * W, 8), 1.0 / 8.0)
                V = np.zeros(H * W)
                population.append((policy, V))
            print(f"population = {population[0]}")
            best_reward_overall = -float('inf')
            final_path_overall = []

            print(f"--- 시작: Actor-Critic + Evolutionary Algorithm (AC-EA) ---")
            print(f"세대 수: {n_generations}, 개체 수: {population_size}, 변이율: {mutation_rate}")

            for gen in range(n_generations):
                fitness_scores = []

                all_paths = []
                for i in range(population_size):
                    policy_, V_ = population[i]
                    if self.inherit == "darwin": 
                        policy = policy_.copy()
                        V = V_.copy()
                    else:
                        policy = policy_    
                        V = V_

                    x, y = self.init_p()
                    state = self.pos_to_idx(x, y, W)
                    total_reward = 0
                    path = [(x, y)]
                    heatmap_values_copy = self.heatmap_values.copy(deep=True)
                    visited = set()
                    visited.add((x, y))

                    for t in range(self.length_episode):
                        action = np.random.choice(len(policy[state]), p=policy[state])
                        # if np.all(policy[state] == 0):
                        #     action = random.choice(range(len(policy[state])))
                        # else:
                        #     if np.random.random() < 0.2:
                        #         action = np.random.choice(len(policy[state]), p=policy[state])
                        #     else:
                        #         action = np.argmax(policy[state])

                        dx, dy = self.actions[action]
                        nx, ny = x + dx, y + dy

                        if not self.is_valid(nx, ny, H, W):
                            nx, ny = x, y

                        if (nx, ny) in visited:
                            candidates = []

                            for alt_action in range(8):
                                adx, ady = self.actions[alt_action]
                                ax, ay = x + adx, y + ady
                                if self.is_valid(ax, ay, H, W) and (ax, ay) not in visited:
                                    candidates.append((alt_action, ax, ay))

                            # if candidates:
                            #     alt_action, ax, ay = random.choice(candidates)
                            #     nx, ny = ax, ay
                            #     action = alt_action
                            if candidates:
                                # 후보지들 중 Heatmap 값(보상)이 가장 높은 곳을 선택 (탐욕적 회피)
                                # 또는 policy[state] 확률이 가장 높은 곳을 선택해도 됨
                                best_cand = None
                                max_cand_val = -float('inf')

                                for cand in candidates:
                                    c_act, c_x, c_y = cand
                                    # 지금 당장 먹을 수 있는 점수를 확인
                                    val = heatmap_values_copy.iloc[c_x, c_y] 
                                    if val > max_cand_val:
                                        max_cand_val = val
                                        best_cand = cand
                                
                                # 만약 모든 후보지의 가치가 같다면(0이라면) 그냥 아무거나
                                if best_cand is None:
                                    best_cand = random.choice(candidates)

                                alt_action, ax, ay = best_cand
                                nx, ny = ax, ay
                                action = alt_action

                        reward = heatmap_values_copy.iloc[nx, ny]
                        coverage = 1 - np.exp(-(H * W / (self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time'])))
                        heatmap_values_copy.iloc[nx, ny] = int(reward * (1 - coverage))

                        next_state = self.pos_to_idx(nx, ny, W)
                        td_error = reward + gamma * V[next_state] - V[state]

                        V[state] += alpha * td_error
                        policy[state, action] += beta * td_error
                        policy[state] = np.maximum(policy[state], 1e-8)
                        policy[state] /= np.sum(policy[state])

                        x, y = nx, ny
                        state = next_state
                        path.append((x, y))
                        visited.add((x, y))
                        total_reward += reward

                    fitness_scores.append(total_reward)
                    all_paths.append(path)
                    if total_reward > best_reward_overall:
                        best_reward_overall = total_reward
                        final_path_overall = path

                avg_fitness = np.mean(fitness_scores)
                max_fitness = np.max(fitness_scores)
                print(
                    f"[세대 {gen + 1}/{n_generations}] 평균 적합도: {avg_fitness:.2f}, 최고 적합도: {max_fitness:.2f}, (전체 최고: {best_reward_overall:.2f})")
                utils.print_and_log(logger, f"fitness = {best_reward_overall:.2f}")

                new_population = []
                num_elites = int(self.pop_size * 0.2)
                elite_indices = np.argsort(fitness_scores)[-num_elites:]

                for idx in elite_indices:
                    elite_policy = population[idx][0].copy()
                    elite_critic = population[idx][1].copy()
                    new_population.append((elite_policy, elite_critic))

                parent_indices = list(range(self.pop_size))
                random.shuffle(parent_indices)
                idx_pointer = 0
                ############랜덤샐랙션#################
                ####################################
                # while len(new_population) < population_size:
                #     if idx_pointer >= population_size - 1:
                #         random.shuffle(parent_indices)
                #         idx_pointer = 0
                #
                #     p1_idx = parent_indices[idx_pointer]
                #     p2_idx = parent_indices[idx_pointer + 1]
                #     parent1 = population[p1_idx]
                #     parent2 = population[p2_idx]


                ############토너먼트샐랙션#################
                ####################################
                while len(new_population) < population_size:
                    # 부모 1: 랜덤 5명 뽑아서 그 중 1등 선택 (경쟁)
                    cands1 = random.sample(range(population_size), 5)
                    p1_idx = max(cands1, key=lambda i: fitness_scores[i])
                    # print(f"p1 = {p1_idx}")
                    # 부모 2: 랜덤 5명 뽑아서 그 중 1등 선택
                    cands2 = random.sample(range(population_size), 5)
                    p2_idx = max(cands2, key=lambda i: fitness_scores[i])
                    # print(f"p2 = {p2_idx}")
                    # (선택) 엄마 아빠 같으면 다시 뽑기
                    if p1_idx == p2_idx:
                        cands2 = random.sample(range(population_size), 5)
                        p2_idx = max(cands2, key=lambda i: fitness_scores[i])
                    # print(f"p1 = {p1_idx}, p2 = {p2_idx}")

                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]
                    if self.crossover_mode == "quarter":
                        child1 = self._crossover(parent1, parent2, H, W)
                    elif self.crossover_mode == "row":
                        child1 = self._crossover_row(parent1, parent2, H, W)
                    elif self.crossover_mode == "col":
                        child1 = self._crossover_col(parent1, parent2, H, W)
                    self._mutate(child1, mutation_rate, H, W)

                    if gen > 0: 
                        self._apply_local_search(child1, H, W, strength=self.strength)

                    if len(new_population) < population_size:
                        new_population.append(child1)
                    idx_pointer += 2
                # center 8389
                population = new_population
            print(f"--- 진화 완료. 최종 최고 보상: {best_reward_overall} ---")
            print(f"final_path = {final_path_overall}")

            final_path_po = []

            for i in range(len(final_path_overall)):
                final_path_po.append(self.output_df.iloc[final_path_overall[i]])
            print(f"final_path_po = {final_path_po}")

            # cpp_formatted_str = self.format_path_for_cpp(final_path_po)
            # print(f"cpp_formatted_str = {cpp_formatted_str}")
            return final_path_po, best_reward_overall
    
    def _apply_local_search(self, individual, H, W, strength=0.05):
        """
        수정된 Local Search:
        단순히 히트맵(보상)이 높은 곳이 아니라,
        Critic(V)이 판단하기에 '가치가 높은 상태'로 이동할 확률을 높임.
        """
        policy, critic = individual # 개체에서 정책과 크리틱을 분리
        
        # V 값을 격자 형태로 변환하여 주변 탐색 용이하게 함
        value_grid = critic.reshape(H, W)

        for r in range(H):
            for c in range(W):
                state_idx = self.pos_to_idx(r, c, W)
                
                best_action = -1
                max_val = -float('inf')
                
                # 주변 8방향 중 V(State Value)가 가장 높은 곳을 찾음
                # 즉, "장기적으로 봤을 때 좋은 곳"을 찾음
                for action_idx, (dr, dc) in enumerate(self.actions):
                    nr, nc = r + dr, c + dc
                    
                    if self.is_valid(nr, nc, H, W):
                        # 중요: 히트맵 값(Reward)이 아니라 학습된 가치(Value)를 사용
                        val = value_grid[nr, nc] 
                        if val > max_val:
                            max_val = val
                            best_action = action_idx
                
                # 해당 방향의 확률을 높여줌 (Lamarckian Learning)
                if best_action != -1:
                    probs = policy[state_idx]
                    
                    # 단순히 더하는 것이 아니라, 비율을 고려하여 안정적으로 증가
                    # 기존 확률 분포를 깨뜨리지 않으면서 가이드만 제공
                    probs[best_action] += strength
                    
                    # 확률 재정규화
                    probs = np.maximum(probs, 1e-8) # 0이 되는 것 방지
                    policy[state_idx] = probs / np.sum(probs)
                    
        return (policy, critic)
    
    def format_path_for_cpp(self, final_path):
        """
        ['129/35', '130/36'] 같은 문자열 리스트를
        C++ 입력용 "0,129,35,130,36..." 문자열로 변환
        """
        # 1. C++ 식별용 인덱스 0 추가
        csv_list = ["0"] 

        # 2. 반복문 수정: 변수 2개(x,y) 대신 1개(point_str)로 받음
        for point_str in final_path:
            # point_str은 현재 "129.123/35.456" 같은 문자열 상태입니다.
            
            # 슬래시(/)를 기준으로 문자열을 쪼갭니다.
            # 예: "129/35".split('/') -> ["129", "35"]
            if '/' in point_str:
                coords = point_str.split('/')
                lon = coords[0].strip() # 공백제거
                lat = coords[1].strip() # 공백제거
                
                csv_list.append(lon)
                csv_list.append(lat)
            else:
                print(f"포맷 에러: {point_str} 안에 '/'가 없습니다.")

        # 3. 쉼표로 이어 붙여서 리턴
        return ",".join(csv_list)