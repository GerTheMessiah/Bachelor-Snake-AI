import os
from collections import ChainMap
from itertools import chain
from statistics import mean

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
from pandas import DataFrame


from src.common.utils import save_path


def make_statistics(agent: dict, agent_color: dict, x_label: str, y_label: str, FIG_PATH: str, x_value_offset: int):
    plt.figure(figsize=(16, 6))
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    handles = []
    sorted(agent)
    for i, (key, value) in enumerate(agent.items()):
        length = len(value)
        plt.plot(list(range(x_value_offset, length + x_value_offset)), value, linewidth=1, color=agent_color[key])
        handles.append(mlines.Line2D([], [], color=agent_color[key], markersize=30, label=key))

    plt.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.2))
    if x_value_offset != 0:
        plt.xticks(range(x_value_offset, x_value_offset + length))
    plt.savefig(FIG_PATH, bbox_inches='tight')


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


def join_agent_dictionaries(STATISTIC_RUN_NUMBER_LIST: list, RUN_TYPE: str, USE_CASE: str, AGENT_LIST: list):
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
        print(key, "&", round(tmp_df["wins"].mean(), 4), "&", round(tmp_df["wins"].std(), 4), r"\\", "\n\hline")
        agent_dict[key] = tmp_df
        if USE_CASE == "test":
            tmp_df = DataFrame()
            agent_dict_rgs[key] = tmp_df.append([df[key] for df in agent_dict_list])
    return agent_dict, agent_dict_rgs, agent_color


def generate_statistic(agent_dict: dict, agent_dict_rgs: dict, agent_color: dict, MODEL_DIR_PATH: str, USE_CASE: str):
    performance_dict = {key: value["apples"].rolling(200).mean().fillna(0) for key, value in agent_dict.items()}
    win_dict = {key: value["wins"].rolling(200).mean().fillna(0) for key, value in agent_dict.items()}
    efficiency_dict = {}
    for key, value in agent_dict.items():
        eff_list = []
        for i in range(64):
            eff_list.append((value[value["apples"].eq(i)]["steps"]).mean())
        efficiency_dict[key] = eff_list
    robustness_dict = {}

    for key, value in agent_dict_rgs.items():
        value["robustheit"] = value["apples-rgs"] / (value["play_ground_size"] ** 2)
        t = [value[value["play_ground_size"].eq(k)]["robustheit"].mean() for k in range(6, 11)]
        robustness_dict[key] = t

    make_statistics(performance_dict, agent_color, "Epochs", "Durchschnittliche Performance der letzten 200 Epochs pro Epoch",
                    MODEL_DIR_PATH + "\\performance-rate.png", 0)

    make_statistics(efficiency_dict, agent_color, "Apfelanzahl", "Effizienz pro Apfelanzahl",
                    MODEL_DIR_PATH + "\\effizienz-rate.png", 0)
    if USE_CASE == "test":
        make_statistics(robustness_dict, agent_color, "Spielfeldgröße", "Durchschnittliche Performance pro Spielfeldgröße",
                        MODEL_DIR_PATH + "\\robustheit-rate.png", 6)

    make_statistics(win_dict, agent_color, "Epochs", "Durchschnittliche Sieg-Rate der letzten 200 Epochs pro Epoch",
                    MODEL_DIR_PATH + "\\win-rate.png", 0)


if __name__ == '__main__':
    path_ = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\statistic"
    USE_CASE = "test"
    agent_dict, agent_dict_rgs, agent_color = join_agent_dictionaries([10, 11], "optimized", USE_CASE, [])
    generate_statistic(agent_dict, agent_dict_rgs, agent_color, path_, USE_CASE)
