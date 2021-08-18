from src.snakeAI.agents.ppo.ppo_train import train_play_ppo
from multiprocessing import Pool

if __name__ == '__main__':
    params1 = {"N_ITERATIONS": 30000, "LR_ACTOR": 0.15e-3, "LR_CRITIC": 0.4e-3, "GAMMA": 0.99, "GPU": True,
               "K_EPOCHS": 10, "EPS_CLIP": 0.2, "BOARD_SIZE": (8, 8)}

    params2 = {"N_ITERATIONS": 30000, "LR_ACTOR": 0.15e-3, "LR_CRITIC": 0.4e-3, "GAMMA": 0.95, "GPU": True,
               "K_EPOCHS": 10, "EPS_CLIP": 0.2, "BOARD_SIZE": (8, 8)}

    params3 = {"N_ITERATIONS": 30000, "LR_ACTOR": 0.15e-3, "LR_CRITIC": 0.4e-3, "GAMMA": 0.90, "GPU": True,
               "K_EPOCHS": 10, "EPS_CLIP": 0.2, "BOARD_SIZE": (8, 8)}

    with Pool(processes=2) as w:
        a = [params1, params1, params2, params2, params3, params3]
        results = [w.apply_async(train_play_ppo, kwds=params) for params in a]
        w.close()
        w.join()
