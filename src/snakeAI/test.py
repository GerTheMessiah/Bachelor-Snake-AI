import os
from statistics import mean, variance, pvariance, stdev
import pandas as pd

from src.common.utils import get_random_game_size


def fun1():
    a = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\baseline-run-01\PPO-01-rgs-test.csv"
    b = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\baseline-run-02\PPO-01-rgs-test.csv"
    a_df = pd.read_csv(a, index_col=0).sort_values(["play_ground_size"])
    b_df = pd.read_csv(b, index_col=0).sort_values(["play_ground_size"])
    a_df["robustheit"] = a_df["apples"] / (a_df["play_ground_size"] ** 2)
    b_df["robustheit"] = b_df["apples"] / (b_df["play_ground_size"] ** 2)
    print(a_df[a_df["play_ground_size"].eq(10)]["robustheit"].mean())
    print(a_df[a_df["play_ground_size"].eq(10)].mean())
    # res = pd.concat([a_df, b_df], axis=1)
    # res["steps"] = res.iloc[:, [1, 7]].mean(axis=1)
    # res["apples"] = res.iloc[:, [2, 8]].mean(axis=1)
    #
    #
    # # print((res["apples"] / a))
    # res["score"] = res.iloc[:, [3, 9]].mean(axis=1)
    # res["wins"] = res.iloc[:, [4, 10]].mean(axis=1)
    # res["apples_new"] = res.iloc[:, [2, 8]].mean(axis=1)
    #
    # res["apples"].rolling(100).mean().fillna(0)


if __name__ == '__main__':

    a = rf"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\optimized-run-02\PPO-03-opt-b-test.csv"
    b = rf"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\optimized-run-02\PPO-03-rgs-opt-b-test.csv"
    a_df = pd.read_csv(a, index_col=0)
    b_df = pd.read_csv(b, index_col=0)
    res = a_df
    res["apples-rgs"] = b_df["apples"]
    res["play_ground_size"] = b_df["play_ground_size"]
    res = res.reset_index(drop=True)
    print(res)
    res.to_csv(a)
    # res.to_csv(b + ".csv")
    # print(res)
    # a = r"C:\Users\Lorenz Mumm\PycharmProjects\Bachelor-Snake-AI\src\resources\optimized-run-02\PPO-03-opt-b-test.csv"
    # y = pd.read_csv(a, index_col=0)
    # #t = y[y["play_ground_size"].eq(10)]["apples"].mean()
    # s = y["apples"].mean()
    # var = y["apples"].var()

    # eff = (y["apples"] / y["steps"]).mean()
    # print(s)
