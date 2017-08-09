import sys
import os
import numpy as np

ant_trace = 1
decay = 0.9
min_pheromone = 1
number_of_tries = 1000
output_param = "./out.txt"
param1 = 60
number_of_ants = 10
param3 = 500


# change system TODO
def freeze(density):
    """Calculate the time for the ant to wait, this simulates road/arc length"""
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

    # Add pheromone trace by ants current position
    for ant in ants:
        pher_mat[ant.current] += ant_trace

    # Apply pheromone decay and pheromone lower threshold
    pher_mat *= decay
    pher_mat[pher_mat < min_pheromone] = min_pheromone

    return create_boarders(pher_mat)


def init_pheromone_mat(shape):
    """Initialize pheromone network, with dimensions specified by shape, shape must be tuple"""
    pher_mat = np.empty(shape)
    pher_mat[:, :, :] = min_pheromone

    return create_boarders(pher_mat)


def create_ants(n, start_point, end_point, mat, path_length, start_id):
    """Initialize n new ants"""
    ants = []
    for i in range(n):
        ant = Ant(mat, start_id + i, path_length, start_point, end_point)
        ants.append(ant)

    return ants


def create_distribution(current, mat):
    """Create new distribution over the available moves for the current position
    according to the pheromone network"""
    p = np.ndarray(shape=[6])
    move_space = []
    choices = [(-1, 0, 0), (1, 0, 0),
               (0, -1, 0), (0, 1, 0),
               (0, 0, -1), (0, 0, 1)]

    for i in range(6):
        c = choices[i]
        p[i] = mat[tuple(np.add(current, c))]
        move_space.append(tuple(np.add(current, c)))

    return move_space, p / p.sum()


def get_valid_solutions(ants, path_length, end_point):
    """Filter out invalid solutions"""
    return filter(lambda ant: ant.get_path_length() == path_length and ant.current == end_point, ants)


def path_to_string(ant):
    """Return string representing the path"""
    path = str(ant.path[0])
    for v in ant.path[1:]:
        path += ' --> ' + str(v)
    return path


class Ant:
    """This class implements ant agents for Ant colony optimization"""

    def __init__(self, mat, id_num, path_length, start_point, end_point):
        # TODO I think {mat} should not be an attribute but rather be passed
        # as an argument to functions using it. This can reduce memory use
        # and make the code more clear, this is not a private property of the ant.
        self.mat = mat
        self.id = id_num
        self.path = [start_point]
        self.current = start_point
        self.score = 0  # TODO should we add the starting point score?
        self.timeout = 0  # TODO should we consider the starting point density?
        self.alive = True
        self.ttl = path_length
        self.end_point = end_point

    def __str__(self):
        """Return basic info about the ant"""
        ant_info = "Ant number " + str(self.id) + '\n'
        ant_info += "Is alive : " + str(self.alive) + '\n'
        ant_info += "Current score: " + str(self.score) + '\n'
        ant_info += "Current path is: " + path_to_string(self) + '\n'
        return ant_info

    def __repr__(self):
        ant_repr  = "Ant([\n"
        ant_repr += " Mat=" + repr(self.mat) + '\n'
        ant_repr += " Id=" + repr(self.id) + '\n'
        ant_repr += " Path=" + repr(self.path) + '\n'
        ant_repr += " Current point=" + repr(self.current) + '\n'
        ant_repr += " Score=" + repr(self.score) + '\n'
        ant_repr += " Timeout=" + repr(self.timeout) + '\n'
        ant_repr += " Alive=" + repr(self.alive) + '\n'
        ant_repr += " Time to live=" + repr(self.ttl) + '\n'
        ant_repr += " End point=" + repr(self.end_point) + '\n'
        ant_repr += "])"
        return ant_repr

    def get_path_length(self):
        """Return the length of the current path found by the ant"""
        return len(self.path) - 1

    def move(self, pheromone_mat):
        """Chose the next step randomly"""
        # Check if the ant is alive
        if not self.alive:
            return

        # Check if the ant is not frozen
        if self.timeout == 0:

            # Create distribution over the valid moves
            move_space, p = create_distribution(self.current, pheromone_mat)
            tries = number_of_tries
            while tries > 0:

                # Choose next move according to the distribution p
                next_move = move_space[np.random.choice(6, p=p)]

                # Check for loops in the path, if loop not found, append new move
                if next_move not in self.path:
                    self.current = next_move
                    self.path.append(next_move)
                    self.score += self.mat[next_move]
                    self.ttl -= 1
                    break

                tries -= 1

            # Calculate the freeze time for the ant
            self.timeout = freeze(self.mat[self.current])

            # Check ant got to destination
            if self.current == self.end_point and self.ttl == 0:
                print(self)
                print(repr(self))
            # return # TODO uncomment this maybe
            # os.system("echo " + ant_info + " >> " + output_param)
            # print(ant_info)

            # Kill ant if no new move selected or no new move is possiable
            if tries == 0 or self.ttl == 0:
                self.alive = False
                return
        else:
            # Decrease freeze time
            self.timeout -= 1


# TODO what???
def main(em_mat, start_point, end_point, path_length):
    pher_mat = init_pheromone_mat([len(em_mat), len(em_mat[0]), len(em_mat[0][0])])
    ants = []
    for i in range(param1):
        ants += create_ants(number_of_ants, start_point, end_point, em_mat, path_length, len(ants))
        for j in range(param3):
            pher_mat = update(pher_mat, ants)

    print("Best path found:\n" + str(max(get_valid_solutions(ants, path_length, end_point), key=lambda ant: ant.score)))


if __name__ == '__main__':
    em_mat = [
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
         [0.1, 0.2, 0.3, 0.4, 0.5]],
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
         [0.1, 0.2, 0.3, 0.4, 0.5]],
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
         [0.1, 0.2, 0.3, 0.4, 0.5]],
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
         [0.1, 0.2, 0.3, 0.4, 0.5]],
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5],
         [0.1, 0.2, 0.3, 0.4, 0.5]]]
    # mat = np.empty((len(em_mat) + 2, len(em_mat[0]) + 2, len(em_mat[0][0]) + 2))
    mat = np.array(em_mat)
    # print(mat)
    start_point = (1, 1, 1)
    end_point = (3, 3, 3)
    path_length = 8

    main(mat, start_point, end_point, path_length)
