import numpy as np
from numpy import random as rnd

import matplotlib.pyplot as plt
import PheromoneNetwork as pNet

# change system TODO


def freeze(density):
    """
    Calculate the time for the ant to wait, this simulates
    road/arc length
    """
    return int(min(1 / density, 100))


def create_boarders(pher_mat):
    """Create boarders for the pheromone matrix, this will ensure that
    the ant won't go outside the valid move space"""
    pher_mat[0, :, :] = pher_mat[-1, :, :] = 0
    pher_mat[:, 0, :] = pher_mat[:, -1, :] = 0
    pher_mat[:, :, 0] = pher_mat[:, :, -1] = 0
    return pher_mat


def update(pher_mat, ants):
    """Simulate move of all ants and calculate new pheromone network"""
    for ant in ants:
        ant.move(pher_mat)
    pher_mat(ants)


def create_ants(number_of_ants, start_point, end_point, path_length, start_id):
    """Initialize number_of_ants new ants"""
    ants = []
    for i in range(number_of_ants):
        ant = Ant(start_id + i, path_length, start_point, end_point)
        ants.append(ant)

    return ants


def get_valid_solutions(ants):
    """Filter out invalid solutions"""
    return [ant for ant in ants if ant.is_valid()]


def path_to_string(ant):
    """Return string representing the path"""
    path = str(ant.path[0])
    for vertex in ant.path[1:]:
        path += ' --> ' + str(vertex)
    return path


class Ant(object):
    """This class implements ant agents for Ant colony optimization"""

    def __init__(self, id_num, path_length, start_point, end_point,
                 pheromone_trace=1):
        self.id_num = id_num
        self.pheromone_trace = pheromone_trace
        self.path = [start_point]
        self.current = start_point
        self.score = 0
        self.timeout = 0
        self.alive = True
        self.ttl = path_length
        self.end_point = end_point

    def __str__(self):
        """Return basic info about the ant"""
        ant_info = "Ant number " + str(self.id_num) + '\n'
        ant_info += "Is alive : " + str(self.alive) + '\n'
        ant_info += "Current score: " + str(self.score) + '\n'
        ant_info += "Current path is: " + path_to_string(self) + '\n'
        return ant_info

    def __repr__(self):
        ant_repr = "Ant([\n"
        ant_repr += " ID=" + repr(self.id_num) + '\n'
        ant_repr += " Path=" + repr(self.path) + '\n'
        ant_repr += " Current point=" + repr(self.current) + '\n'
        ant_repr += " Score=" + repr(self.score) + '\n'
        ant_repr += " Pheromone trace=" + repr(self.pheromone_trace) + '\n'
        ant_repr += " Timeout=" + repr(self.timeout) + '\n'
        ant_repr += " Alive=" + repr(self.alive) + '\n'
        ant_repr += " Time to live=" + repr(self.ttl) + '\n'
        ant_repr += " End point=" + repr(self.end_point) + '\n'
        ant_repr += "])"
        return ant_repr

    def get_path_length(self):
        """Return the length of the current path found by the ant"""
        return len(self.path) - 1

    def is_valid(self):
        """Return a list of ants with valid solution paths"""
        return self.ttl == 0 and self.current == self.end_point

    def move(self, mat, pher_net, number_of_tries=100):
        """Chose the next step randomly"""
        # Check if the ant is alive
        if not self.alive:
            return

        # Check if the ant is not frozen
        if self.timeout == 0:

            # Create distribution over the valid moves
            move_space, dist = pher_net.create_distribution(self.current)
            tries = number_of_tries
            while tries > 0:

                # Choose next move according to the distribution dist
                next_move = move_space[rnd.choice(6, p=dist)]

                # Check for loops in the path, if loop not found,
                # append new move
                if next_move not in self.path:
                    self.current = next_move
                    self.path.append(next_move)
                    self.score += mat[next_move]
                    self.ttl -= 1
                    break

                tries -= 1

            # Calculate the freeze time for the ant
            self.timeout = freeze(mat[self.current])

            # Check ant got to destination
            if self.current == self.end_point and self.ttl == 0:
                # print(repr(self))
                print(self)

            # Kill ant if no new move selected or no new move is possiable
            if tries == 0 or self.ttl == 0:
                self.alive = False
                return
        else:
            # Decrease freeze time
            self.timeout -= 1

    # def plot(self):
    #     x = max(self.path, lambda v: v[0])[0]
    #     y = max(self.path, lambda v: v[1])[1]
    #     z = max(self.path, lambda v: v[2])[2]


def main():
    """Test unit for Ant and PheromoneNetwork classes"""
    row = [i/10.0 for i in range(10)]
    layer = [list(row)] * 5
    cryo_em = [list(layer)] * 5

    mat = np.array(cryo_em)

    start_point = (1, 1, 1)
    end_point = (3, 3, 3)
    path_length = 8

    # output_param = "./out.txt"
    rounds = 100
    number_of_ants = 10
    moves = 30

    pher_net = pNet.PheromoneNetwork(
        [len(mat), len(mat[0]), len(mat[0][0])])

    print(pher_net)

    ants = []
    for _ in range(rounds):
        ants += create_ants(number_of_ants, start_point,
                            end_point, path_length, len(ants))
        for _ in range(moves):
            for ant in ants:
                ant.move(mat, pher_net)
            pher_net(ants)

    print(pher_net)
    print("Best path found:\n" +
          str(max(get_valid_solutions(ants), key=lambda ant: ant.score)))


if __name__ == '__main__':
    main()
