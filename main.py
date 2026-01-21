import random
import greedy
import argparse
import prepare_data
import utils
import ga
import rl
import time
import os
import subprocess

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mode", type=str, default="rl", help="greedy, ga, ma, ma2, rl")
arg_parser.add_argument("--seed", type=int, default="3", help="seed value for random number default = 2026")
arg_parser.add_argument("--run", type=int, default="1", help="number of runs")
arg_parser.add_argument("--look", type=str, default="3", help="look 3x3 or 5x5")
arg_parser.add_argument("--inherit", type=str, default="lamarckian", help="dawin and lamarckian")
arg_parser.add_argument("--crossover_mode", type=str, default="row", help="row and col and quarter")
arg_parser.add_argument("--RLmode", type=str, default="ac_ea", help="q or ac or drl_ac or ac_ea")
arg_parser.add_argument("--strength", type=float, default="0.5", help="0-1")

#GA
arg_parser.add_argument("--pop_size", type=int, default="200", help="population size")
arg_parser.add_argument("--generation", type=int, default="200", help="number of generations")
arg_parser.add_argument("--position", type=str, default="leftup", help="leftup, leftdown, rightup, rightdown, center")
arg_parser.add_argument("--mutation_rate", type=float, default="0.2", help="mutation_rate")
def log_run_info(logger, args, run):
    utils.print_and_log(logger, "="*50)
    utils.print_and_log(logger, f"\n{'Run:':15}{run}")
    utils.print_and_log(logger, f"{'Mode:':15}{args.mode}")
    utils.print_and_log(logger, f"{'Seed:':15}{args.seed}")
    
    # 추가된 파라미터들
    utils.print_and_log(logger, f"{'Look:':15}{args.look}")
    utils.print_and_log(logger, f"{'Inherit:':15}{args.inherit}")          # 추가됨
    utils.print_and_log(logger, f"{'Crossover:':15}{args.crossover_mode}") # 추가됨
    utils.print_and_log(logger, f"{'RL mode:':15}{args.RLmode}")
    utils.print_and_log(logger, f"{'Strength:':15}{args.strength}")        # 추가됨 (로컬서치 강도 등)

    # GA 관련 파라미터
    utils.print_and_log(logger, f"{'Population:':15}{args.pop_size}")
    utils.print_and_log(logger, f"{'Generation:':15}{args.generation}")
    utils.print_and_log(logger, f"{'Mutation rate:':15}{args.mutation_rate}")
    utils.print_and_log(logger, f"{'Position:':15}{args.position}")
    utils.print_and_log(logger, "="*50)

if __name__ == "__main__":
    arg = arg_parser.parse_args()
    logger = utils.setup_logger(arg)
    log_run_info(logger, arg, arg.run)
    results = []
    all_time = []
    
    for run in range(arg.run):
        SRU, heatmap_values, output_df, possible_length, coverage, prob = prepare_data.prepare_data()
        
        if arg.mode == 'greedy':
            start = time.time()
            utils.print_and_log(logger, f"Run {run+1}")
            greedy_ = greedy.Greedy(heatmap_values, possible_length, output_df, SRU, arg.position, arg.look)
            final_path, num_detect = greedy_.greedy(logger, arg.seed, run)
            end = time.time() - start

            # # 2. C++이 읽을 수 있는 파일(Vertex File)로 저장
            # vertex_file_path = "./temp/current_path_greedy" + str(run+1) + ".csv"
            # with open(vertex_file_path, "w") as f:
            #     f.write(final_path)

            # print(f"C++용 경로 파일 생성 완료: {vertex_file_path}")
            # print(f"내용 예시: {final_path[:50]}...")

            utils.print_and_log(logger, f"CPU time = {end:.4f}") 
            utils.print_and_log(logger, f"final_path = {final_path}")
            all_time.append(end)
            print(final_path)
            utils.draw(final_path, arg, run)
            results.append(num_detect)
            
        elif arg.mode == 'ga':
            start = time.time()
            utils.print_and_log(logger, f"Run {run+1}")
            Genetic = ga.GA(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = Genetic.evolve(logger, arg.seed, run)
            end = time.time() - start

            # # 2. C++이 읽을 수 있는 파일(Vertex File)로 저장
            # vertex_file_path = "./temp/current_path_ga" + "_pop:" + str(arg.pop_size) + "_gen:" + str(arg.generation) + "_prob:" + str(arg.mutation_rate) + "_" + arg.crossover_mode + str(run+1) + ".csv"
            # with open(vertex_file_path, "w") as f:
            #     f.write(final_path)

            # print(f"C++용 경로 파일 생성 완료: {vertex_file_path}")
            # print(f"내용 예시: {final_path[:50]}...")

            utils.print_and_log(logger, f"CPU time = {end:.4f}")
            utils.print_and_log(logger, f"final_path = {final_path}")
            all_time.append(end)
            print(final_path)
            utils.draw(final_path, arg, run)
            results.append(num_detect)
            
        elif arg.mode == 'ma' or arg.mode == 'ma2':
            start = time.time()
            utils.print_and_log(logger, f"Run {run+1}")
            Genetic = ga.GA(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = Genetic.evolve(logger, arg.seed, run)
            end = time.time() - start

            # vertex_file_path = "./temp/current_path_ma" + "_pop:" + str(arg.pop_size) + "_gen:" + str(arg.generation) + "_prob:" + str(arg.mutation_rate) + "_" + arg.crossover_mode + str(run+1) + ".csv"
            # with open(vertex_file_path, "w") as f:
            #     f.write(final_path)

            # print(f"C++용 경로 파일 생성 완료: {vertex_file_path}")
            # print(f"내용 예시: {final_path[:50]}...")

            utils.print_and_log(logger, f"CPU time = {end:.4f}")
            utils.print_and_log(logger, f"final_path = {final_path}")
            all_time.append(end)
            print(final_path)
            utils.draw(final_path, arg, run)
            results.append(num_detect)
            
        elif arg.mode == 'rl':
            start = time.time()
            utils.print_and_log(logger, f"Run {run+1}")
            rl_ = rl.rl(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = rl_.find_path(logger, arg.seed, run)
            end = time.time() - start

            # 2. C++이 읽을 수 있는 파일(Vertex File)로 저장
            # vertex_file_path = "./temp/current_path_rl_" + arg.RLmode + "_" + str(arg.strength) + "_pop:" + str(arg.pop_size) + "_gen:" + str(arg.generation) + "_prob:" + str(arg.mutation_rate) + "_" + arg.crossover_mode + str(run+1) + ".csv"
            # with open(vertex_file_path, "w") as f:
            #     csv_data = ["0"] 
            #     for point_str in final_path:
            #         try:
            #             lon, lat = point_str.split('/')
            #             csv_data.append(lon.strip()) 
            #             csv_data.append(lat.strip())
            #         except ValueError:
            #             print(f"데이터 포맷 경고: {point_str} (x/y 형식이 아님)")
            #             continue

            #     f.write(",".join(csv_data))

            # print(f"C++용 경로 파일 생성 완료: {vertex_file_path}")
            # print(f"내용 예시: {final_path[:50]}...")

            utils.print_and_log(logger, f"CPU time = {end:.4f}")
            utils.print_and_log(logger, f"final_path = {final_path}")
            all_time.append(end)
            print(final_path)
            utils.draw(final_path, arg, run)
            results.append(num_detect)
            
        elif arg.mode == 'drl':
            utils.print_and_log(logger, f"Run {run+1}")
            rl_obj = rl.rl(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = rl_obj.find_path(logger, arg.seed+run)
            utils.draw(final_path, arg)
            results.append(num_detect)
            all_time.append(0.0) 

    utils.print_and_log(logger, "\n" + "="*50)
    utils.print_and_log(logger, " FINAL SUMMARY PER RUN")
    utils.print_and_log(logger, "="*50)
    
    for i in range(len(results)):
        time_str = f"{all_time[i]:.4f}s" if i < len(all_time) else "N/A"
        utils.print_and_log(logger, f"Run {i+1:02d} : Result = {results[i]:.4f} | Time = {time_str}")
        
    utils.print_and_log(logger, "-"*50)
    
    if results:
        avg_res = sum(results) / len(results)
        utils.print_and_log(logger, f"Average Result = {avg_res:.4f}")
    if all_time:
        avg_time = sum(all_time) / len(all_time)
        utils.print_and_log(logger, f"Average Time   = {avg_time:.4f}s")
        
    utils.print_and_log(logger, "="*50)