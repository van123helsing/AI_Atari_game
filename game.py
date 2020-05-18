import gym
import numpy as np
from game_models.ddqn_trainer import DDQNTrainer
from game_models.ddqn_solver import DDQNSolver
from gym_wrappers import MainGymWrapper
import sys
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
            env.render('human')

            action = game_model.move(current_state)
            next_state, reward, done, info = env.step(action)

            # normaliziramo nagrado
            np.sign(reward)
            score += reward

            # shranimo
            tic = time.perf_counter()
            game_model.remember(current_state, action, reward, next_state, done)
            current_state = next_state

            game_model.step_update(total_step)
            toc = time.perf_counter()
            print(format(toc - tic, '.2f'))

            if done:
                game_model.save_run(score, step, run)


def main():
    game_name = "Breakout-v0"
    env = MainGymWrapper.wrap(gym.make(game_name))
    if len(sys.argv) > 1:
        AB = sys.argv[1]
        if AB == 'A':
            _main_loop(DDQNTrainer(game_name, INPUT_SHAPE, env.action_space.n), env)
        elif AB == 'B':
            _main_loop(DDQNSolver(game_name, INPUT_SHAPE, env.action_space.n), env)
        else:
            print("Kot parameter morate vnesti eno izmed črk (A/B)")
    else:
        print("Kot parameter morate vnesti eno izmed črk (A/B)")


if __name__ == "__main__":
    main()
