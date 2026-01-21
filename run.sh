#!/bin/bash

MAX_JOBS=8
count=0

run_job () {
    $1 &
}

commands=(
"python3 main.py --mode greedy --seed 2026 --position leftup --RLmode ac_ea --run 30"
"python3 main.py --mode greedy --seed 2026 --position leftdown --RLmode ac_ea --run 30"
"python3 main.py --mode greedy --seed 2026 --position center --RLmode ac_ea --run 30"
"python3 main.py --mode greedy --seed 2026 --position rightup --RLmode ac_ea --run 30"
"python3 main.py --mode greedy --seed 2026 --position rightdown --RLmode ac_ea --run 30"

"python3 main.py --mode ga --seed 2026 --position leftup --run 30"
"python3 main.py --mode ga --seed 2026 --position leftdown --run 30"
"python3 main.py --mode ga --seed 2026 --position center --run 30"
"python3 main.py --mode ga --seed 2026 --position rightup --run 30"
"python3 main.py --mode ga --seed 2026 --position rightdown --run 30"

"python3 main.py --mode ma --seed 2026 --position leftup --run 30"
"python3 main.py --mode ma --seed 2026 --position leftdown --run 30"
"python3 main.py --mode ma --seed 2026 --position center --run 30"
"python3 main.py --mode ma --seed 2026 --position rightup --run 30"
"python3 main.py --mode ma --seed 2026 --position rightdown --run 30"

"python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac --run 30"
"python3 main.py --mode rl --seed 2026 --position leftdown --RLmode ac --run 30"
"python3 main.py --mode rl --seed 2026 --position center --RLmode ac --run 30"
"python3 main.py --mode rl --seed 2026 --position rightup --RLmode ac --run 30"
"python3 main.py --mode rl --seed 2026 --position rightdown --RLmode ac --run 30"


"python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ea --inherit lamarckian --crossover_mode quarter --run 30"
"python3 main.py --mode rl --seed 2026 --position leftdown --RLmode ac_ea --inherit lamarckian --crossover_mode quarter --run 30"
"python3 main.py --mode rl --seed 2026 --position center --RLmode ac_ea --inherit lamarckian --crossover_mode quarter --run 30"
"python3 main.py --mode rl --seed 2026 --position rightup --RLmode ac_ea --inherit lamarckian --crossover_mode quarter --run 30"
"python3 main.py --mode rl --seed 2026 --position rightdown --RLmode ac_ea --inherit lamarckian --crossover_mode quarter --run 30"

"python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --inherit lamarckian --crossover_mode quarter --run 30 --strength 0.8"
"python3 main.py --mode rl --seed 2026 --position leftdown --RLmode ac_ma --inherit lamarckian --crossover_mode quarter --run 30 --strength 0.8"
"python3 main.py --mode rl --seed 2026 --position center --RLmode ac_ma --inherit lamarckian --crossover_mode quarter --run 30 --strength 0.8"
"python3 main.py --mode rl --seed 2026 --position rightup --RLmode ac_ma --inherit lamarckian --crossover_mode quarter --run 30 --strength 0.8"
"python3 main.py --mode rl --seed 2026 --position rightdown --RLmode ac_ma --inherit lamarckian --crossover_mode quarter --run 30 --strength 0.8"

)

for cmd in "${commands[@]}"; do
    run_job "$cmd"
    ((count++))

    if ((count % MAX_JOBS == 0)); then
        wait
    fi
done

wait
echo "done"


# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 100 --generation 200 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"
# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 100 --generation 400 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"

# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 200 --generation 200 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"
# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 200 --generation 300 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"
# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 200 --generation 400 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"

# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 300 --generation 200 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"
# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 300 --generation 300 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"
# "python3 main.py --mode rl --seed 2026 --position leftup --RLmode ac_ma --pop_size 300 --generation 400 --inherit lamarckian --crossover_mode quarter --run 10 --strength 0.8"