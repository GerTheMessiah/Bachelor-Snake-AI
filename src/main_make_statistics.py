import sys

from src.statistic.statisticTool import get_data, join_agent_dicts, generate_statistic

if __name__ == '__main__':
    args = sys.argv
    FIG_PATH = args[1]
    USE_CASE = str(args[2])
    RUN_TYPE = str(args[3])
    STATISTIC_RUN_NUMBER = int(args[4])
    AGENT_LIST = list(args[5])
    agent_dict_list, agent_dict_rgs_list, agent_color_list = [], [], []

    for i in range(1, STATISTIC_RUN_NUMBER + 1):
        agent_dict, agent_dict_rgs, agent_color = get_data([1], RUN_TYPE, USE_CASE, [])
        agent_dict_list.append(agent_dict)
        agent_dict_rgs_list.append(agent_dict_rgs)
        agent_color_list.append(agent_color)

    len_first = len(agent_color_list[0]) if agent_color_list else 0
    if not all(len(i) == len_first for i in agent_color_list):
        raise ValueError('Not all runs contains the same length!')

    a = join_agent_dicts(agent_dict_list, agent_dict_rgs_list, USE_CASE)
    generate_statistic(*a, agent_color=agent_color_list[0], FIG_PATH=FIG_PATH, USE_CASE=USE_CASE)

