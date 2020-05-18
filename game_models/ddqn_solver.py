import numpy as np
import os
import random
from game_models.base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork


EXPLORATION_TEST = 0.02


class DDQNSolver(BaseGameModel):
    def __init__(self, game_name, input_shape, action_space):

        model_path = "./output/neural_nets/" + game_name + "/ddqn/model.h5"
        logger_path = "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/"

        BaseGameModel.__init__(self, game_name,
                               "DDQN Solver",
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space).model

        if os.path.isfile(self.model_path):
            self.ddqn.load_weights(self.model_path)

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])