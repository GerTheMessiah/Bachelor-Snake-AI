import sys

from src.statistic.statisticTool import load_agents

if __name__ == '__main__':
    args = sys.argv
    params = {"STATISTIC_RUN_NUMBER": int(args[1]), "RUN_TYPE": str(args[2]), "USE_CASE": str(args[3]),
              "AGENT_LIST": list(args[4])}
    load_agents(**params)

