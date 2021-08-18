import os
import re
import sys
from os import listdir
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import linregress


def plot_learning_curve(x: list, apples: list, scores: list, figure_file: str):
    plt.figure()
    fig, axs = plt.subplots(2, 1, figsize=(16, 11))
    axs[0].plot(x, apples, color='red')
    m, b, _, _, _ = linregress(x, apples)
    axs[0].plot(x, [m * y + b for y in x], color='blue')
    axs[0].grid(True)
    axs[0].set_xlabel('Number of Games')
    axs[0].set_ylabel("Sum of Apples per Game")

    axs[1].plot(x, scores, color='green')
    m2, b2, _, _, _ = linregress(x, scores)
    axs[1].plot(x, [m2 * y + b2 for y in x], color='orange')
    axs[1].set_xlabel('Number of Games')
    axs[1].set_ylabel(r"Sum of Scores per Game")
    axs[1].grid(True)
    fig.tight_layout()

    red_patch = mlines.Line2D([], [], color='red', markersize=30, label=f'Apple_max: {max(apples)}')
    blue_patch = mlines.Line2D([], [], color='blue', markersize=30, label=f'Apple_reg m: {round(m, 4)}, b: {round(b, 4)}')
    red2_patch = mlines.Line2D([], [], color='green', markersize=30, label=f'Score_max: {round(max(scores), 2)}')
    blue2_patch = mlines.Line2D([], [], color='orange', markersize=30, label=f'Score_reg m: {round(m2, 4)}, b: {round(b2, 4)}')
    fig.legend(handles=[red_patch, blue_patch, red2_patch, blue2_patch], loc="lower center", ncol=4)
    fig.subplots_adjust(bottom=0.1)

    fig.show()
    fig.savefig(figure_file)


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


def file_path(dir: str, new_save: bool, file_name: str = "model"):
    MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + fr"\resources\{dir}"
    try:
        MODEL_ID = max([int(re.sub(r'\D', '', item)) for item in listdir(MODEL_DIR_PATH)])
    except Exception:
        print("Loading model failed")
        return rf"{MODEL_DIR_PATH}\{file_name}_0"
    return rf"{MODEL_DIR_PATH}\{file_name}_{MODEL_ID + 1 if new_save else MODEL_ID}"


def save_file(path, **kwargs) -> str:

    MODEL_DIR_PATH = str(Path(__file__).parent.parent.parent.parent) + "\\resources\\" if not path else path
    DIR_NAME = "Save-"
    for parameter, value in kwargs.items():

        DIR_NAME += (str(parameter).lower() + "_" + str(value) + "-")

    DIR_NAME = DIR_NAME[:-1]

    os.mkdir(MODEL_DIR_PATH + DIR_NAME)

    return MODEL_DIR_PATH + DIR_NAME
