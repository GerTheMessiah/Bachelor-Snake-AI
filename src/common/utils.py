import os
import sys
from pathlib import Path
from random import randint
from typing import Tuple
from pandas import DataFrame

""" 
Source of the Method:
https://www.programcreek.com/python/
?code=Azure%2Fazure-diskinspect-service%2Fazure-diskinspect-service-master%2FpyServer%2FAzureDiskInspectService.py
"""


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\r\n')
    sys.stdout.flush()





def save_path(statistic_run_number: int, alg_type: str, agent_number: int, use_case: str, run_type: str,
              random_game_size: bool, optimization) -> Tuple[str, str]:
    """
    @params:
    statistic_run_number - defines the name of the run director \n
    alg_type - algorithm type - differentiate between DQN and PPO \n
    agent_number - number of the agent. Important for saving files. \n
    use_case - differentiate between test or train run. Important for naming the files \n
    run_type - differentiate between baseline or optimized run \n
    random_game_size - was a random game size used. Important for naming files. \n
    optimization - which optimization was used. "A", "B", "None"
    """
    if not (run_type == "baseline" or run_type == "optimized"):
        raise ValueError("Wrong RUN_TYPE!")
    if not (alg_type == "PPO" or alg_type == "DQN"):
        raise ValueError("Wrong ALG_TYPE!")
    if not (use_case == "train" or use_case == "test"):
        raise ValueError("Wrong USE_CASE!")
    if not (optimization is "A" or optimization is "B" or optimization is None):
        raise ValueError("Wrong optimization! Use A, B or None")

    MODEL_DIR_PATH = str(Path(__file__).parent.parent) + f"\\resources\\{run_type}-run-0{statistic_run_number}"
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)

    if optimization is None:
        opt = ""
    else:
        opt = f"opt-{optimization.lower()}-"

    if random_game_size:
        rgs = MODEL_DIR_PATH + f"\\{alg_type}-0{agent_number}-rgs-{opt}{use_case}"
    else:
        rgs = MODEL_DIR_PATH + f"\\{alg_type}-0{agent_number}-{opt}{use_case}"

    return rgs, MODEL_DIR_PATH


def save(alg_type, agent_number, statistic_run_number, use_case, run_type, random_game_size, agent, dtime, steps_list,
         apples, scores, wins, play_ground_size=None, optimization=None):
    if play_ground_size is None:
        play_ground_size = []
    path_, _ = save_path(statistic_run_number=statistic_run_number, alg_type=alg_type, agent_number=agent_number,
                         use_case=use_case, run_type=run_type, random_game_size=random_game_size,
                         optimization=optimization)
    if use_case == "train":
        agent.store_model(PATH=path_ + ".model")
    if not random_game_size:
        df = DataFrame({"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins})
    else:
        df = DataFrame({"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins,
                        "play_ground_size": [i[0] for i in play_ground_size]})
    df.to_csv(path_ + ".csv")
    print(path_)


def get_random_game_size(epoch, div=1000, offset=6):
    a = offset + (epoch // div)
    return a, a
