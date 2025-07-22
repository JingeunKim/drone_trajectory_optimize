import random
import greedy
import argparse
import prepare_data
import utils
import ga
import rl

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mode", type=str, default="ma", help="greedy, ga, ma, ma2, rl")
arg_parser.add_argument("--seed", type=int, default="3", help="seed value for random number default = 2026")
arg_parser.add_argument("--run", type=int, default="1", help="number of runs")
arg_parser.add_argument("--look", type=str, default="3", help="look 3x3 or 5x5")

#GA
arg_parser.add_argument("--pop_size", type=int, default="100", help="population size")
arg_parser.add_argument("--generation", type=int, default="300", help="number of generations")
arg_parser.add_argument("--position", type=str, default="leftdown", help="leftup, leftdown, rightup, rightdown, center")
arg_parser.add_argument("--mutation_rate", type=float, default="0.2", help="mutation_rate")

def log_run_info(logger, args, run):
    utils.print_and_log(logger, "="*50)
    utils.print_and_log(logger, f"\n{'Run:':15}{run}")
    utils.print_and_log(logger, f"{'Mode:':15}{args.mode}")
    utils.print_and_log(logger, f"{'Seed:':15}{args.seed}")
    utils.print_and_log(logger, f"{'Population size:':15}{args.pop_size}")
    utils.print_and_log(logger, f"{'Generation:':15}{args.generation}")
    utils.print_and_log(logger, f"{'Mutation rate:':15}{args.mutation_rate}")
    utils.print_and_log(logger, f"{'Position:':15}{args.position}")
    utils.print_and_log(logger, f"{'look:':15}{args.look}")
    utils.print_and_log(logger, "="*50)

if __name__ == "__main__":
    arg = arg_parser.parse_args()
    logger = utils.setup_logger(arg)
    log_run_info(logger, arg, arg.run)
    results = []
    for run in range(arg.run):
        SRU, heatmap_values, output_df, possible_length, coverage, prob = prepare_data.prepare_data()
        if arg.mode == 'greedy':
            utils.print_and_log(logger, f"Run {run+1}")
            greedy_ = greedy.Greedy(heatmap_values, possible_length, output_df, SRU, arg.position, arg.look)
            final_path, num_detect = greedy_.greedy(logger, arg.seed+run)
            print(final_path)
            utils.draw(final_path, arg)
            results.append(num_detect)
        elif arg.mode == 'ga':
            utils.print_and_log(logger, f"Run {run+1}")
            Genetic = ga.GA(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = Genetic.evolve(logger, arg.seed+run)
            utils.draw(final_path, arg)
            results.append(num_detect)
        elif arg.mode == 'ma' or arg.mode == 'ma2':
            utils.print_and_log(logger, f"Run {run+1}")
            Genetic = ga.GA(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = Genetic.evolve(logger, arg.seed+run)
            utils.draw(final_path, arg)
            results.append(num_detect)
        elif arg.mode == 'rl': #15:greedy 20:GA 20:Greedy 20:GA
            utils.print_and_log(logger, f"Run {run+1}")
            rl = rl.rl(heatmap_values, possible_length, output_df, SRU, arg)
            final_path, num_detect = rl.find_path(logger, arg.seed+run)
            # Genetic = ga.GA(heatmap_values, possible_length, output_df, SRU, arg)
            # final_path, num_detect = Genetic.evolve(logger, arg.seed+run)
            utils.draw(final_path, arg)
            results.append(num_detect)
        elif arg.mode == 'rl':
            pass
        elif arg.mode == 'nco':
            pass
    utils.print_and_log(logger, f"average result = {sum(results) / len(results)}")