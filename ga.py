import random
import utils
import numpy as np
import copy


class GA:
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

    def initialization(self):
        population = []
        for i in range(self.pop_size):
            population.append(self.generate_population())
        return population

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(self.heatmap_values) and 0 <= ny < len(self.heatmap_values[0]):
                    neighbors.append((nx, ny))
        return neighbors


    def generate_population(self, greedy=False):
        pop = []
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
        pop.append((x, y))
        # print(f"self.heatmap_values = {len(self.heatmap_values)}")
        for _ in range(self.possible_length - 1):
            neighbors = self.get_neighbors(x, y)

            x, y = random.choice(neighbors)
            pop.append((x, y))
        return pop

    def convert_path(self, individual):
        path = []
        path.append(individual[0])
        init = individual[0]
        for n in range(1, len(individual)):
            x_y = individual[n]
            x_new = init[0] + x_y[0]
            y_new = init[1] + x_y[1]
            init = (x_new, y_new)
            path.append(init)
        return path

    def fitness(self, individual, total_coverage):
        # print("fitness")
        # print(f"individual = {individual}")
        # print(f"individual[0] = {individual[0]}")
        # path = self.convert_path(individual)
        # print(f"path = {path}")

        detect_point = 0
        heatmap_values_copy = copy.deepcopy(self.heatmap_values)
        # print("fitness")
        # print(f"self.possible_length - 1 = {self.possible_length - 1}")
        # print(f"heatmap_values_copy= {heatmap_values_copy}")
        for i in range(self.possible_length - 1):
            # print(f"heatmap_values_copy.iloc[individual[i]] = {heatmap_values_copy.iloc[individual[i]]}")
            actual_coverage = self.SRU['speed'] * self.SRU['coverage'] * i * self.SRU['time']
            coverage = 1 - np.exp(-(total_coverage / actual_coverage))
            # print(f"individual[i] = {individual[i]}")
            # print(f"heatmap_values_copy = {heatmap_values_copy}")
            # print(f"heatmap_values_copy.iloc[individual[i]] = {heatmap_values_copy.iloc[individual[i]]}")
            # print(f"heatmap_values_copy = {heatmap_values_copy}")
            detect_point += heatmap_values_copy.iloc[individual[i]]
            heatmap_values_copy.iloc[individual[i]] = int(heatmap_values_copy.iloc[individual[i]] * (1 - coverage))

        return detect_point

    def reconstruct_path(self, start, diffs):
        path = [start]
        current = start
        for dx, dy in diffs:  # len(self.heatmap_values)
            x = current[0] + dx
            y = current[1] + dy
            if x >= len(self.heatmap_values):
                x = len(self.heatmap_values) - 1
            elif x <= 0:
                x = 0

            if y >= len(self.heatmap_values):
                y = len(self.heatmap_values) - 1
            elif y <= 0:
                y = 0
            current = (x, y)
            path.append(current)
        return path

    def crossover(self, individual1, individual2):
        pt = random.randint(0, len(individual1) - 1)

        diffs1 = []
        diffs2 = []
        diffs1.append((0, 0))
        diffs2.append((0, 0))
        for i in range(1, len(individual1)):
            x1, y1 = individual1[i - 1]
            x2, y2 = individual1[i]
            diffs1.append((x2 - x1, y2 - y1))

            x1, y1 = individual2[i - 1]
            x2, y2 = individual2[i]
            diffs2.append((x2 - x1, y2 - y1))

        diffs11 = diffs1[:pt] + diffs2[pt:]
        diffs22 = diffs2[:pt] + diffs1[pt:]

        offspring1 = self.reconstruct_path(individual1[0], diffs11[1:])
        offspring2 = self.reconstruct_path(individual2[0], diffs22[1:])

        # offspring1 = individual1[:pt] + individual2[pt:]
        # offspring2 = individual2[:pt] + individual1[pt:]

        return offspring1, offspring2

    def mutation(self, individual1, individual2):
        offspring1 = []
        offspring2 = []
        # print("individual1:", individual1)
        # print("individual2:", individual2)
        pt = random.randint(0, len(individual1) - 1)

        diffs1 = []
        diffs2 = []
        diffs1.append((0, 0))
        diffs2.append((0, 0))
        for i in range(1, len(individual1)):
            x1, y1 = individual1[i - 1]
            x2, y2 = individual1[i]
            diffs1.append((x2 - x1, y2 - y1))

            x1, y1 = individual2[i - 1]
            x2, y2 = individual2[i]
            diffs2.append((x2 - x1, y2 - y1))

        rd = [-1, 0, 1]
        dx_old, dy_old = diffs1[pt]

        # 모든 조합 중에서 현재 변화량 제외
        candidates = [(dx, dy) for dx in rd for dy in rd if (dx, dy) != (dx_old, dy_old)]
        new = random.choice(candidates)
        diffs1[pt] = new

        dx_old2, dy_old2 = diffs2[pt]

        # 모든 조합 중에서 현재 변화량 제외
        candidates2 = [(dx, dy) for dx in rd for dy in rd if (dx, dy) != (dy_old2, dy_old2)]
        new2 = random.choice(candidates2)
        diffs2[pt] = new2

        offspring1 = self.reconstruct_path(individual1[0], diffs1[1:])
        offspring2 = self.reconstruct_path(individual2[0], diffs2[1:])
        # print("offspring1:", offspring1)
        # print("offspring2:", offspring2)
        return offspring1, offspring2

    def selection(self, selected_number):
        parents = random.sample(selected_number, 2)
        for rand in parents:
            selected_number.remove(rand)
        return parents, selected_number

    def move_to_local_optimum(self, path, start_idx, rows, cols, max_iterations=5):
        # print(f"original path = {path}")
        # print(f"start_idx = {start_idx}")

        for idx in range(start_idx, len(path)):
            iterations = 0
            while iterations < max_iterations:
                current_x, current_y = path[idx]
                best_x, best_y = current_x, current_y
                best_value = self.heatmap_values.iloc[current_x, current_y]

                if self.look == '3':
                    for i in range(current_x - 1, current_x + 2):
                        for j in range(current_y - 1, current_y + 2):
                            if 0 <= i < rows and 0 <= j < cols:
                                value = self.heatmap_values.iloc[i, j]
                                if value > best_value:
                                    best_value = value
                                    best_x, best_y = i, j
                elif self.look == '5':
                    for dx in [-1, 0, 1]:
                        # print(f"current_x = {current_x}, current_y = {current_y}")
                        for dy in [-1, 0, 1]:
                            i1 = current_x + dx
                            j1 = current_y + dy
                            i2 = current_x + 2 * dx
                            j2 = current_y + 2 * dy
                            # print(f"i1 = {i1}, j1 = {j1}, i2 = {i2}, j2 = {j2}")
                            val1 = 0
                            val2 = 0

                            if 0 <= i1 < rows and 0 <= j1 < cols:
                                val1 = self.heatmap_values.iloc[i1, j1]
                                if val1 == 0:
                                    val1 += 1
                                # print(f"val1 = {val1}")
                            if 0 <= i2 < rows and 0 <= j2 < cols:
                                val2 = self.heatmap_values.iloc[i2, j2]
                                if val2 == 0:
                                    val2 += 1
                                # print(f"val2 = {val2}")

                            score = 0.8 * val1 + 0.2 * val2

                            if score > best_value:
                                best_value = score
                                best_x, best_y = i1, j1
                    # for i in range(current_x - 1, current_x + 4):
                    #     for j in range(current_y - 1, current_y + 4):
                    #         if 0 <= i < rows and 0 <= j < cols:
                    #             value = self.heatmap_values.iloc[i, j]
                    #             if value > best_value:
                    #                 best_value = value
                    #                 best_x, best_y = i, j
                path[idx] = (best_x, best_y)
                iterations += 1

        return path

    def local(self, offspring1, offspring2):
        pt1 = random.randint(1, len(offspring1) - 1)
        rows = len(self.heatmap_values)
        cols = len(self.heatmap_values.columns)

        offspring1 = self.move_to_local_optimum(offspring1, pt1, rows, cols)
        offspring2 = self.move_to_local_optimum(offspring2, pt1, rows, cols)

        return offspring1, offspring2

    def local2(self, offspring1, offspring2):
        print(f"self.heatmap_values = {self.heatmap_values}")
        print(f"offspring1 = {offspring1}")
        print(f"offspring1[0] = {offspring1[0]}")
        num = []
        for i in range(len(offspring1)):
            aaa = self.heatmap_values.iloc[offspring1[i]]
            num.append(aaa)
        print(f"num = {num}, min = {min(num)}, max = {max(num)}")
        return offspring1, offspring2

    def group_consecutive_points(self, points):
        if not points:
            return []

        points.sort()
        intervals = []
        start = points[0]
        prev = points[0]

        for p in points[1:]:
            if p == prev + 1:
                prev = p
            else:
                intervals.append((start, prev))
                start = p
                prev = p

        # 마지막 구간 추가
        intervals.append((start, prev))
        return intervals

    def find_path(self, intervals, offspring):
        for j in range(len(intervals)):
            pt = intervals[j]
            init = pt[0] - 1
            if pt[1] + 1 >= len(offspring):
                continue
            goal = pt[1] + 1
            diff = pt[1] - pt[0] + 1

            start_coord = offspring[init]
            goal_coord = offspring[goal]

            best_path = []
            best_sum = -1
            visited = set()
            max_attempts = 100
            attempts = [0]

            def dfs(path, steps, total_particles):
                nonlocal best_path, best_sum
                if attempts[0] >= max_attempts:
                    return

                attempts[0] += 1
                current = path[-1]

                if steps == diff:
                    if abs(current[0] - goal_coord[0]) <= 1 and abs(current[1] - goal_coord[1]) <= 1:
                        if total_particles > best_sum:
                            best_sum = total_particles
                            best_path = path.copy()
                    return

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = current[0] + dx, current[1] + dy

                        if 0 <= nx < self.heatmap_values.shape[0] and 0 <= ny < self.heatmap_values.shape[1]:
                            if (nx, ny) not in visited:
                                visited.add((nx, ny))
                                path.append((nx, ny))
                                particles = self.heatmap_values.iloc[nx, ny]
                                dfs(path, steps + 1, total_particles + particles)
                                path.pop()
                                visited.remove((nx, ny))

            visited.add(start_coord)
            start_particles = self.heatmap_values.iloc[start_coord[0], start_coord[1]]
            dfs([start_coord], 0, start_particles)

            # print(f"Best path found: {best_path}")

            if len(best_path) == diff:
                for k in range(len(best_path)):
                    offspring[init + k + 1] = best_path[k]
        return offspring
    def repair(self, offspring1, offspring2):
        # print("=" * 50)
        # print("repair phase")
        # print(f"offspring1 = {offspring1}")
        # print(f"offspring2 = {offspring2}")
        # print(self.heatmap_values)

        points = []
        for i in range(len(offspring2)):
            if self.heatmap_values.iloc[offspring2[i]] == 0:
                points.append(i)

        # print(f"points = {points}")
        intervals = self.group_consecutive_points(points)
        # print(f"intervals = {intervals}")
        offspring1 = self.find_path(intervals, offspring1)
        offspring2 = self.find_path(intervals, offspring2)

        return offspring1, offspring2

    def evolve(self, logger, seed):
        total_coverage = len(self.heatmap_values) * len(self.heatmap_values)

        random.seed(seed)
        genetic_path = []

        population = self.initialization()
        # print(f"population: {population}")
        fitness_values = []
        # print("init")
        # for q in population:
        #     print(q)
        #     print(len(q))

        for i in range(self.pop_size):
            fitness_values.append(self.fitness(population[i], total_coverage))
        print(fitness_values)
        fitness_best = []
        fitness_avg = []
        for g in range(self.generations):
            utils.print_and_log(logger, f"Generation {g + 1}")
            selected_number = list(range(self.pop_size))
            for j in range(int(self.pop_size / 2)):
                pair, selected_number = self.selection(selected_number)
                # print(pair[0], pair[1])
                # print(population[pair[0]])
                # print(population[pair[1]])
                offspring1, offspring2 = self.crossover(population[pair[0]], population[pair[1]])
                indx = random.random()
                if indx > self.muation_rate:
                    offspring1, offspring2 = self.mutation(offspring1, offspring2)
                # print("+"*30)
                # print(f"before offspring1 = {offspring1}")
                offspring1, offspring2 = self.repair(offspring1, offspring2)
                # print(f"after offspring1 = {offspring1}")
                if self.local_search == "ma":
                    offspring1, offspring2 = self.local(offspring1, offspring2)
                elif self.local_search == "ma2":
                    offspring1, offspring2 = self.local2(offspring1, offspring2)

                # print(offspring2)
                population.append(offspring1)
                population.append(offspring2)
                # for p in population:
                #     print(p)
                fitness_values.append(self.fitness(offspring1, total_coverage))
                fitness_values.append(self.fitness(offspring2, total_coverage))

            fitness_rank = np.argsort(fitness_values)[::-1]
            fitness_values = [fitness_values[i] for i in fitness_rank]
            fitness_values = fitness_values[:self.pop_size]
            population = [population[i] for i in fitness_rank]
            # print("before = ", len(population))
            population = population[:self.pop_size]
            # print("after = ", len(population))
            fitness_best.append(fitness_values[0])
            fitness_avg.append(sum(fitness_values) / len(fitness_values))
            # print(f"Population: {population[0]}")
            utils.print_and_log(logger, "Best fitness: {}".format(fitness_values[0]))
            # print("all fitness = ", fitness_values)
            # print("ggggg")
            # for p in population:
            #     print(p)
        final_path = []
        best_individual = population[0]
        # best_individual = self.convert_path(best_individual)
        for i in range(len(best_individual)):
            final_path.append(self.output_df.iloc[best_individual[i]])
        utils.print_and_log(logger, "best path {}".format(final_path))
        utils.print_and_log(logger, "Best fitness {}".format(fitness_values[0]))
        # for p in population:
        #     print(p)
        print("all fitness = ", fitness_values)
        print("final_path = ", final_path)
        return final_path, fitness_values[0]
