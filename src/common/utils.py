import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pandas import DataFrame


def plot_learning_curve(x: list, apples: list, scores: list, figure_file: str):
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(16, 11))
    axs[0].plot(x, apples, color='red')
    axs[0].grid(True)
    axs[0].set_xlabel('Number of Games')
    axs[0].set_ylabel("Sum of Apples per Game")

    axs[1].plot(x, scores, color='green')
    axs[1].set_xlabel('Number of Games')
    axs[1].set_ylabel(r"Sum of Scores per Game")
    axs[1].grid(True)
    fig.tight_layout()

    red_patch = mlines.Line2D([], [], color='red', markersize=30, label=f'Apple_max: {max(apples)}')
    blue_patch = mlines.Line2D([], [], color='blue', markersize=30,
                               label=f'Apple_reg m: {round(m, 4)}, b: {round(b, 4)}')
    red2_patch = mlines.Line2D([], [], color='green', markersize=30, label=f'Score_max: {round(max(scores), 2)}')
    blue2_patch = mlines.Line2D([], [], color='orange', markersize=30,
                                label=f'Score_reg m: {round(m2, 4)}, b: {round(b2, 4)}')
    fig.legend(handles=[red_patch, blue_patch, red2_patch, blue2_patch], loc="lower center", ncol=4)
    fig.subplots_adjust(bottom=0.1)

    fig.show()
    fig.savefig(figure_file)


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


def save_path(statistic_run_number: int, alg_type: str, agent_number: int, use_case: str) -> str:
    MODEL_DIR_PATH = str(
        Path(__file__).parent.parent.parent.parent) + f"\\resources\\statistic-run-0{statistic_run_number}"
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    return MODEL_DIR_PATH + f"\\{alg_type}-0{agent_number}-{use_case}"


def save(alg_type, agent_number, statistic_run_number, use_case, agent, dtime, steps_list, apples, scores, wins):
    path_ = save_path(statistic_run_number=statistic_run_number, alg_type=alg_type, agent_number=agent_number,
                      use_case=use_case)
    if use_case == "train":
        agent.store_model(path=path_ + ".model")
    df = DataFrame({"time": dtime, "steps": steps_list, "apples": apples, "scores": scores, "wins": wins})
    df.to_csv(path_ + ".csv")


