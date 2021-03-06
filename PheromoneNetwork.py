import numpy as np


def create_boarders(pher_net):
    """
    Create boarders for the pheromone matrix, this will ensure that
    the ant won't go outside the valid move space
    """

    pher_net[0, :, :] = pher_net[-1, :, :] = 0
    pher_net[:, 0, :] = pher_net[:, -1, :] = 0
    pher_net[:, :, 0] = pher_net[:, :, -1] = 0

    for helix in pher_net.helicis:
        for voxel in helix:
            pher_net[voxel] = 0

    return pher_net


class PheromoneNetwork(object):
    """
    This Class impliments simple pheromone network
    which simulates ant pheromone
    """

    def __init__(self, shape, helicis, decay=0.9, min_value=1):
        # Inialize pheromone network
        self.net = np.empty(shape)
        self.net[:, :, :] = min_value
        self.net = create_boarders(self.net)

        # Set helicis in map
        self.helicis = helicis

        # set constants
        self.decay = decay
        self.min_value = min_value

    def __call__(self, ants, side_effect=0.2):
        # Add pheromone trace by ants current position
        sides = [(-1, 0, 0), (1, 0, 0),
                 (0, -1, 0), (0, 1, 0),
                 (0, 0, -1), (0, 0, 1)]

        for ant in ants:
            # Check if ant found path
            if ant.current == ant.end_point:  # and ant.ttl == 0:
                # TODO Does any path is good?
                # or just if the ant is with ttl = 0?
                self.net[ant.current] += ant.pheromone_trace
                for side in sides:
                    self.net[ant.current] += ant.pheromone_trace * side_effect

        # Apply pheromone decay and pheromone lower threshold
        self.net *= self.decay
        # Fix the pheromone network
        self.net[(self.net < self.min_value)] = self.min_value
        self.net = create_boarders(self.net)

    def __str__(self):
        return 'Decay : ' + str(self.decay) + '\n' + 'Net :\n' + str(self.net)

    def create_distribution(self, current):
        """
        Create a new distribution over the available moves for the current
        position according to the pheromone network
        """

        prob = np.ndarray(shape=[6])
        move_space = []
        choices = [(-1, 0, 0), (1, 0, 0),
                   (0, -1, 0), (0, 1, 0),
                   (0, 0, -1), (0, 0, 1)]

        for i in range(6):
            prob[i] = self.net[tuple(np.add(current, choices[i]))]
            move_space.append(tuple(np.add(current, choices[i])))

        return move_space, prob / prob.sum()

    def add_pheromone_path(self, path, pheromone_trace):
        """Add pheromone path to the pheromone network"""
        for node in path:
            self.net[node] += pheromone_trace
