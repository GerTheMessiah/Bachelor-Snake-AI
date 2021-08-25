import os

import pandas as pd
from matplotlib import pyplot as plt

from src.common import save_path


def make_statistics(length: int, agent_names: list, x_label: str, y_label: str, *args):
    plt.figure(figsize=(16, 6))
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

    for i, value in enumerate(args):
        plt.plot([range(length)], value, color=color_list[i])

    handles = []
    for i, value in enumerate(agent_names):
        handles.append([], [], color=color_list[i], markersize=30, label=value)

    plt.legend(handles=handles, loc="lower center", ncol=len(agent_names))
    plt.show()
    plt.savefig('plot.fig')


if __name__ == '__main__':
    path_ = save_path(statistic_run_number=1, alg_type="PPO", agent_number=1, use_case="train")
    csv_files = []
    train_df = []
    play_df = []
    directory = os.path.join(path_)
    for _, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(file)

    for file in csv_files:
        if file is not None and "train" in file:
            train_df.append(pd.read_csv(path_ + "-train.csv"))

        if file is not None and "test" in file:
            train_df.append(pd.read_csv(path_ + "\\test.csv"))



