import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from drl_model import ActorCriticAgent
from torch.distributions import Categorical
import torch.optim as optim
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 딥러닝 모델 정의 (Actor-Critic 통합)
class ACNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ACNetwork, self).__init__()
        # 입력: (x, y) 좌표 -> 은닉층 -> 출력
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Actor Head: 행동 확률 출력 (Softmax)
        self.actor = nn.Linear(64, action_dim)

        # Critic Head: 상태 가치 출력 (Scalar)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # 행동 확률 (0~1 사이 값, 합은 1)
        probs = torch.softmax(self.actor(x), dim=-1)
        # 가치 (예측 점수)
        value = self.critic(x)
        return probs, value


# 유전 알고리즘용: 가중치 섞기 (Crossover)
def neural_crossover(parent1_net, parent2_net):
    child_net = ACNetwork(2, 8)  # 새 자식 생성
    child_dict = child_net.state_dict()
    p1_dict = parent1_net.state_dict()
    p2_dict = parent2_net.state_dict()

    # 부모 가중치를 50:50으로 섞음 (평균)
    for key in child_dict:
        child_dict[key] = (p1_dict[key] + p2_dict[key]) / 2.0

    child_net.load_state_dict(child_dict)
    return child_net


# 유전 알고리즘용: 가중치 변이 (Mutation)
def neural_mutate(network, mutation_rate, noise_std=0.1):
    if random.random() < mutation_rate:
        with torch.no_grad():
            for param in network.parameters():
                # 가중치에 노이즈 추가
                noise = torch.randn_like(param) * noise_std
                param.add_(noise)


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

    def find_path(self, logger, seed):
        random.seed(seed)
        np.random.seed(seed)
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

                        if candidates:
                            alt_action, ax, ay = random.choice(candidates)
                            nx, ny = ax, ay
                            a = alt_action

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

                        if candidates:
                            alt_action, ax, ay = random.choice(candidates)
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

            print(f"--- 🧬 시작: Actor-Critic + Evolutionary Algorithm (AC-EA) ---")
            print(f"세대 수: {n_generations}, 개체 수: {population_size}, 변이율: {mutation_rate}")

            for gen in range(n_generations):
                fitness_scores = []

                all_paths = []
                for i in range(population_size):
                    policy, V = population[i]
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

                            if candidates:
                                alt_action, ax, ay = random.choice(candidates)
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
                    child1 = self._crossover(parent1, parent2, H, W)
                    self._mutate(child1, mutation_rate, H, W)
                    if len(new_population) < population_size:
                        new_population.append(child1)
                    idx_pointer += 2
                # center 8389
                population = new_population
            print(f"--- 🧬 진화 완료. 최종 최고 보상: {best_reward_overall} ---")
            print(f"final_path = {final_path_overall}")

            final_path_po = []

            for i in range(len(final_path_overall)):
                final_path_po.append(self.output_df.iloc[final_path_overall[i]])

            return final_path_po, best_reward_overall

        elif self.mode == 'ac_dea':
            H, W = self.heatmap_values.shape
            population_size = self.pop_size
            n_generations = self.generations

            lr = 0.001
            gamma = 0.99

            population = []
            optimizers = []

            for _ in range(population_size):
                net = ACNetwork(input_dim=2, action_dim=8)
                optimizer = optim.Adam(net.parameters(), lr=lr)
                population.append(net)
                optimizers.append(optimizer)

            best_reward_overall = -float('inf')
            final_path_overall = []

            print(f"--- 🧬 시작: Deep Neuro-Evolution (AC-EA) ---")

            for gen in range(n_generations):
                print(f"generation = {gen}")
                fitness_scores = []
                all_paths = []

                print(f"pop = ")
                for i in range(population_size):
                    net = population[i]
                    optimizer = optimizers[i]

                    x, y = self.init_p()
                    total_reward = 0
                    path = [(x, y)]
                    heatmap_values_copy = self.heatmap_values.copy(deep=True)
                    visited = set()
                    visited.add((x, y))
                    print(f"0000000")

                    for t in range(self.length_episode):
                        print(f"asdfasfsdf")
                        state_tensor = torch.FloatTensor([x / H, y / W]).unsqueeze(0)

                        print(f"net = {net}")
                        probs, value = net(state_tensor)
                        print(f"state_tensor = {state_tensor}")
                        print(f"probs = {probs}")
                        probs = probs.squeeze(0)
                        value = value.squeeze(0)

                        dist = torch.distributions.Categorical(probs)
                        action_tensor = dist.sample()
                        action = action_tensor.item()
                        log_prob = dist.log_prob(action_tensor)
                        print(f"log_prob = {log_prob}")

                        dx, dy = self.actions[action]
                        nx, ny = x + dx, y + dy

                        if not self.is_valid(nx, ny, H, W):
                            nx, ny = x, y

                        if (nx, ny) in visited:
                            candidates = []
                            for alt_act in range(8):
                                adx, ady = self.actions[alt_act]
                                ax, ay = x + adx, y + ady
                                if self.is_valid(ax, ay, H, W) and (ax, ay) not in visited:
                                    candidates.append((alt_act, ax, ay))
                            if candidates:
                                alt_action, ax, ay = random.choice(candidates)
                                nx, ny = ax, ay
                                action = alt_action
                            else:
                                nx, ny = x, y

                                # 보상 계산
                        reward = heatmap_values_copy.iloc[nx, ny]
                        print(f"99999")
                        print(f"{reward}")
                        scaled_reward = reward / 100.0  #
                        print(f"99999")
                        print(f"{scaled_reward}")

                        coverage = 1 - np.exp(-(H * W / (self.SRU['speed'] * self.SRU['coverage'] * self.SRU['time'])))
                        heatmap_values_copy.iloc[nx, ny] = int(reward * (1 - coverage))
                        print(f"99999")

                        next_state_tensor = torch.FloatTensor([nx / H, ny / W]).unsqueeze(0)
                        _, next_value = net(next_state_tensor)
                        next_value = next_value.squeeze(0)
                        print(f"8888")
                        print(f"next_value = {next_value}")

                        td_target = scaled_reward + gamma * next_value.detach()
                        td_error = td_target - value
                        print(f"scaled_reward = {scaled_reward}")
                        print(f"next_value = {next_value}")
                        print(f"value = {value}")
                        print(f"td_error = {td_error}")
                        print(f"777777")

                        critic_loss = td_error.pow(2)
                        actor_loss = -log_prob * td_error.detach()
                        print(f"critic_loss = {critic_loss}")
                        print(f"log_prob = {log_prob}")
                        print(f"actor_loss = {actor_loss}")
                        print(f"666666")

                        entropy = dist.entropy()
                        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                        print(f"55555")
                        episode_loss = episode_loss + loss



                        optimizer.zero_grad()
                        print(f"4")
                        loss.backward()
                        print(f"3")
                        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                        print(f"1")

                        optimizer.step()
                        print(f"0")

                        # 상태 갱신
                        x, y = nx, ny
                        path.append((x, y))
                        visited.add((x, y))
                        total_reward += reward

                    fitness_scores.append(total_reward)
                    all_paths.append(path)
                    print(f"3333")

                    if total_reward > best_reward_overall:
                        best_reward_overall = total_reward
                        final_path_overall = path
                    print(f"22222")

                avg_fitness = np.mean(fitness_scores)
                max_fitness = np.max(fitness_scores)
                print(
                    f"[세대 {gen + 1}/{n_generations}] 평균: {avg_fitness:.2f}, 최고: {max_fitness:.2f},")

                # --- [3] 다음 세대 생성 (Neuro-Evolution) ---
                new_population = []
                new_optimizers = []

                # 1. 엘리트 보존
                num_elites = int(self.pop_size*0.2)
                elite_indices = np.argsort(fitness_scores)[-num_elites:]
                for idx in elite_indices:
                    # 신경망 복제 (Deep Copy)
                    parent_net = population[idx]
                    child_net = ACNetwork(2, 8)
                    child_net.load_state_dict(parent_net.state_dict())

                    new_population.append(child_net)
                    new_optimizers.append(optim.Adam(child_net.parameters(), lr=lr))

                # 2. 토너먼트 선택 및 교차
                while len(new_population) < population_size:
                    # 부모 선택 (Tournament)
                    cands1 = random.sample(range(population_size), 5)
                    p1_idx = max(cands1, key=lambda i: fitness_scores[i])
                    cands2 = random.sample(range(population_size), 5)
                    p2_idx = max(cands2, key=lambda i: fitness_scores[i])

                    parent1 = population[p1_idx]
                    parent2 = population[p2_idx]

                    # 신경망 교차 (Crossover)
                    child_net = neural_crossover(parent1, parent2)

                    # 신경망 변이 (Mutation)
                    neural_mutate(child_net, self.muation_rate)

                    new_population.append(child_net)
                    new_optimizers.append(optim.Adam(child_net.parameters(), lr=lr))

                population = new_population
                optimizers = new_optimizers

            print(f"--- 🧬 Deep Neuro-Evolution 완료. 최종 최고 보상: {best_reward_overall} ---")

            final_path_po = []
            for i in range(len(final_path_overall)):
                final_path_po.append(self.output_df.iloc[final_path_overall[i]])

            return final_path_po, best_reward_overall

    def visualize_q_table(self, Q, H, W):
        best_actions = np.argmax(Q, axis=1)  # 각 상태별 가장 큰 값의 index (최적 행동)
        best_actions_2D = best_actions.reshape(H, W)  # 환경의 크기로 재배열

        plt.figure(figsize=(10, 8))
        sns.heatmap(best_actions_2D, annot=True, cmap="coolwarm", cbar=True, square=True)
        plt.title("Q-table Best Action per State (Reshaped)")
        plt.xlabel("Y")
        plt.ylabel("X")
        plt.show()
