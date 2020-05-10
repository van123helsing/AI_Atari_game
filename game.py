import gym
import numpy as np
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from gym_wrappers import MainGymWrapper
import time

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


def _main_loop(game_model, env):

    run = 0
    total_step = 0
    while True:
        run += 1
        current_state = env.reset()
        step = 0
        score = 0
        done = False

        while not done:
            if total_step >= 5000000:
                print("Reached total step limit.")
                exit(0)
            total_step += 1
            step += 1

            # graificni prikaz
            # env.render()

            action = game_model.move(current_state)
            next_state, reward, done, info = env.step(action)

            # normaliziramo nagrado
            np.sign(reward)
            score += reward
            # shranimo
			
            game_model.remember(current_state, action, reward, next_state, done)
            current_state = next_state

            tic = time.perf_counter()
            game_model.step_update(total_step)
            toc = time.perf_counter()
            print(format(toc - tic, '.2f'))

            if done:
                game_model.save_run(score, step, run)


def main():
    game_name = "Breakout-v0"
    env = MainGymWrapper.wrap(gym.make(game_name))
    _main_loop(DDQNTrainer(game_name, INPUT_SHAPE, env.action_space.n), env)

    # DDQNSolver(game_name, INPUT_SHAPE, action_space)


if __name__ == "__main__":
    main()