import os
from itertools import chain
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from pandas import DataFrame


from src.common.utils import save_path

"""
This method plots and saves the input data for each agent.
@:param agent: Python dict with the agent name as key and a list with the data to be plotted as value.
@:param agent_color: Python dict with the agent name as key and the agent color als value.
@:param x_label: Label for the x axis.
@:param y_label: Label for the y axis.
@:param FIG_PATH: Path for saving the statistic.
@:param x_value_offset: Offset for the x axis. Normally starting at 0.
"""
def make_statistics(agent: dict, agent_color: dict, x_label: str, y_label: str, FIG_PATH: str, x_value_offset: int,
                    marker: str):
    plt.figure(figsize=(16, 6))
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    handles = []
    sorted(agent)
    for i, (key, value) in enumerate(agent.items()):
        length = len(value)
        plt.plot(list(range(x_value_offset, length + x_value_offset)), value, linewidth=1, color=agent_color[key],
                 marker=marker)
        handles.append(mlines.Line2D([], [], color=agent_color[key], markersize=30, label=key))

    plt.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.2))
    if x_value_offset != 0:
        plt.xticks(range(x_value_offset, x_value_offset + length))
    plt.savefig(FIG_PATH, bbox_inches='tight', format='svg')

"""
Method for loading the csv files in which the data of the agents is saved.
@:param STATISTIC_RUN_NUMBER: Number of statistic run. Important for the saving and loading path.
@:param RUN_TYPE: "baseline" or "optimized"
@:param USE_CASE: "train" or "test"
@:return df_dict: Python dict with agent name as key and Dataframe with the csv data as value.
"""
def load_agents(STATISTIC_RUN_NUMBER: int, RUN_TYPE: str, USE_CASE: str):
    _, MODEL_DIR_PATH = save_path(statistic_run_number=STATISTIC_RUN_NUMBER, alg_type="PPO", agent_number=1,
                                  random_game_size=False, use_case=USE_CASE, run_type=RUN_TYPE, optimization=None)

    directory = os.path.join(MODEL_DIR_PATH)
    df_dict = {}
    for _, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv") and USE_CASE in file and "opt" not in file:
                df_dict[file[:6]] = pd.read_csv(MODEL_DIR_PATH + "\\" + file, index_col=0)
            elif file.endswith(".csv") and USE_CASE in file and "opt" in file:
                df_dict[str(file[:6] + file[6:12])] = pd.read_csv(MODEL_DIR_PATH + "\\" + file, index_col=0)

    return df_dict

"""
This method joins the agent dicts and returns the agents dictionaries with the data of the evaluation criteria for each 
agent.
@:param agent_dicts_list: List of the agents with the normal game size (8x8)
@:param agent_dicts_rgs_list: List of the agents with the random game size.
@:param USE_CASE: Test run or train run?
"""
def join_agent_dicts(agent_dicts_list: list, agent_dicts_rgs_list: list, USE_CASE: str):
    performance_list, win_list, efficiency_list, robustness_list, agent_name_list = [], [], [], [], []
    for agent_dict in agent_dicts_list:
        agent_name_list.append(list(agent_dict.keys()))
        performance_list.append({key: value["apples"].rolling(200).mean().fillna(0).tolist() for key, value in agent_dict.items()})
        win_list.append({key: value["wins"].rolling(200).mean().fillna(0).tolist() for key, value in agent_dict.items()})
        efficiency_dict = {}
        for key, value in agent_dict.items():
            eff_list = []
            for i in range(64):
                eff_list.append((value[value["apples"].eq(i)]["steps"]).mean())
            efficiency_dict[key] = eff_list
        efficiency_list.append(efficiency_dict)
    if USE_CASE == "test":
        for agent_dict_rgs in agent_dicts_rgs_list:
            robustness_dict = {}
            for key, value in agent_dict_rgs.items():
                value["robustheit"] = value["apples-rgs"] / (value["play_ground_size"] ** 2)
                t = [value[value["play_ground_size"].eq(k)]["robustheit"].mean().tolist() for k in range(6, 11)]
                robustness_dict[key] = t
            robustness_list.append(robustness_dict)

    performance_dict, win_dict, efficiency_dict, robustness_dict = {}, {}, {}, {}
    for agent_name in sorted(set(list(chain(*list(agent_name_list))))):
        tmp = [list(agent[agent_name]) for agent in performance_list]
        arrays = [np.array(x) for x in tmp]
        performance_dict[agent_name] = [np.mean(k) for k in zip(*arrays)]
        tmp = [list(agent[agent_name]) for agent in win_list]
        arrays = [np.array(x) for x in tmp]
        win_dict[agent_name] = [np.mean(k) for k in zip(*arrays)]
        tmp = [list(agent[agent_name]) for agent in efficiency_list]
        arrays = [np.array(x) for x in tmp]
        efficiency_dict[agent_name] = [np.nanmean(k) for k in zip(*arrays)]
        if USE_CASE == "test":
            tmp = [list(agent[agent_name]) for agent in robustness_list]
            arrays = [np.array(x) for x in tmp]
            robustness_dict[agent_name] = [np.nanmean(k) for k in zip(*arrays)]
    return performance_dict, win_dict, efficiency_dict, robustness_dict

"""
This method loads the data from csv files.
@:param STATISTIC_RUN_NUMBER: Number of statistic run. Important for the saving and loading path.
@:param RUN_TYPE: "baseline" or "optimized"
@:param USE_CASE: "train" or "test"
@:param Agent_LIST: Regulates the returned data of the agents. No data will be processed for agents which are in the
 agent list.
"""
def get_data(STATISTIC_RUN_NUMBER_LIST: list, RUN_TYPE: str, USE_CASE: str, AGENT_LIST: list):
    agent_dict_list, agent_name_list, agent_color_name_list, agent_color = [], [], [], {}
    color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#0f2911', '#0d1975', '#ffff33', '#a65628', '#f781bf', '#888888',
                  '#6699CC']
    for statistic_run_number in STATISTIC_RUN_NUMBER_LIST:
        agent_dict = load_agents(statistic_run_number, RUN_TYPE, USE_CASE)
        agent_color_name_list.append(list(agent_dict.keys()))
        for agent in AGENT_LIST:
            agent_dict.pop(agent, None)
        agent_name_list.append(list(agent_dict.keys()))
        agent_dict_list.append(agent_dict)
    agent_name_list = sorted(list(set(list(chain(*list(agent_name_list))))))
    for i, name in enumerate(sorted(list(set(chain(*list(agent_color_name_list)))))):
        agent_color[name] = color_list[i]
    agent_dict, agent_dict_rgs = {}, {}
    for key in agent_name_list:
        tmp_df = DataFrame()
        res = pd.concat([df[key] for df in agent_dict_list], axis=1)
        tmp_df["steps"] = res.iloc[:, [i for i, x in enumerate(res.columns) if x == "steps"]].mean(axis=1)
        tmp_df["apples"] = res.iloc[:, [i for i, x in enumerate(res.columns) if x == "apples"]].mean(axis=1)
        tmp_df["score"] = res.iloc[:, [i for i, x in enumerate(res.columns) if x == "score"]].mean(axis=1)
        tmp_df["wins"] = res.iloc[:, [i for i, x in enumerate(res.columns) if x == "wins"]].mean(axis=1)
        # print(key, "&", round(tmp_df["steps"].mean(), 4), "&", round(tmp_df["steps"].std(), 4), r"\\", "\n\hline")
        agent_dict[key] = tmp_df
        if USE_CASE == "test":
            tmp_df = DataFrame()
            agent_dict_rgs[key] = tmp_df.append([df[key] for df in agent_dict_list])

    return agent_dict, agent_dict_rgs, agent_color

"""
This method creates the statistics for each evaluation criteria.
@:param performance_dict: Dictionary with the data of the performance for each agent.
@:param win_dict: Dictionary with the data of the win rate for each agent.
@:param efficiency_dict: Dictionary with the data of the efficiency for each agent.
@:param robustness_dict: Dictionary with the data of the robustness for each agent.
@:param agent_color: Dictionary with agents as key and the hex color as value.
@:param FIG_PATH: Saving path of the statistic.
@:param USE_CASE: "train" or "test".
"""
def generate_statistic(performance_dict: dict, win_dict: dict, efficiency_dict: dict, robustness_dict: dict,
                       agent_color: dict, FIG_PATH: str, USE_CASE: str):

    make_statistics(performance_dict, agent_color, "Epoch", "Durchschn. Apfelanzahl der letzten 200 Epochs",
                    FIG_PATH + "\\performance_rate.svg", 0, marker='')

    make_statistics(efficiency_dict, agent_color, "Apfelanzahl", "Durchschn. Schrittanzahl",
                    FIG_PATH + "\\effizienz_rate.svg", 0, marker='.')
    if USE_CASE == "test":
        make_statistics(robustness_dict, agent_color, r"$\sqrt{Spielfeldgröße}$", "Durchschn. Apfelanzahl / Spielfeldgröße",
                        FIG_PATH + "\\robustheit_rate.svg", 6, marker='.')

    make_statistics(win_dict, agent_color, "Epoch", "Durchschn. Siegrate der letzten 200 Epochs",
                    FIG_PATH + "\\win_rate.svg", 0, marker='')


if __name__ == '__main__':
    path_ = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\statistic"
    USE_CASE = "test"
    RUN_TYPE = "baseline"
    RUNS = 2
    agent_dict_list, agent_dict_rgs_list, agent_color_list = [], [], []

    for i in range(1, RUNS + 1):
        agent_dict, agent_dict_rgs, agent_color = get_data([1], RUN_TYPE, USE_CASE, [])
        agent_color_list.append(agent_dict)
        agent_dict_rgs_list.append(agent_dict_rgs)
        agent_color_list.append(agent_color)
    agent_color = {}
    it = iter(agent_dict_list)
    the_len = len(next(it))
    if not all(len(l) == the_len for l in it):
        raise ValueError('not all lists have same length!')
    a = join_agent_dicts(agent_dict_list, agent_dict_rgs_list, USE_CASE)
    generate_statistic(*a, agent_color=agent_color_list[0], FIG_PATH=path_, USE_CASE=USE_CASE)
