import os
from pathlib import Path


if __name__ == '__main__':
    dir_path = str(Path(__file__).parent.parent.parent.parent) + "\\statistics\\"
    number_of_dir = len(next(os.walk(dir_path))[1])
    os.mkdir(dir_path + f"statistic-run-{number_of_dir}")
