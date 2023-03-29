from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import trange

from openelm.environments import BaseEnvironment, Genotype

Phenotype = Optional[np.ndarray]
MapIndex = Optional[tuple]


class Map:
    """
    Class to represent a map of any dimensionality, backed by a numpy array.

    This class is necessary to handle the circular buffer for the history dimension.
    """

    def __init__(
        self,
        dims: tuple,
        fill_value: float,
        dtype: type = np.float32,
        history_length: int = 1,
    ):
        """
        Class to represent a map of any dimensionality, backed by a numpy array.

        This class is a wrapper around a numpy array that handles the circular
        buffer for the history dimension. We use it for the array of fitnesses,
        genomes, and to track whether a niche in the map has been explored or not.

        Args:
            dims (tuple): Tuple of ints representing the dimensions of the map.
            fill_value (float): Fill value to initialize the array.
            dtype (type, optional): Type to pass to the numpy array to initialize
                it. For example, we initialize the map of genomes with type `object`.
                Defaults to np.float32.
            history_length (int, optional): Length of history to store for each
                niche (cell) in the map. This acts as a circular buffer, so after
                storing `history_length` items, the buffer starts overwriting the
                oldest items. Defaults to 1.
        """
        self.history_length: int = history_length
        self.dims: tuple = dims
        if self.history_length == 1:
            self.array: np.ndarray = np.full(dims, fill_value, dtype=dtype)
        else:
            # Set starting top of buffer to 0 (% operator)
            self.top = np.full(dims, self.history_length - 1, dtype=int)
            self.array = np.full((history_length,) + dims, fill_value, dtype=dtype)
        self.empty = True

    def __getitem__(self, map_ix):
        """If history length > 1, the history dim is an n-dim circular buffer."""
        if self.history_length == 1:
            return self.array[map_ix]
        else:
            return self.array[(self.top[map_ix], *map_ix)]

    def __setitem__(self, map_ix, value):
        self.empty = False
        if self.history_length == 1:
            self.array[map_ix] = value
        else:
            top_val = self.top[map_ix]
            top_val = (top_val + 1) % self.history_length
            self.top[map_ix] = top_val
            self.array[(self.top[map_ix], *map_ix)] = value

    @property
    def shape(self) -> tuple:
        """Wrapper around the shape of the numpy array."""
        return self.array.shape

    @property
    def map_size(self) -> int:
        """Returns the product of the dimension sizes, not including history."""
        if self.history_length == 1:
            return self.array.size
        else:
            return self.array[0].size

    @property
    def qd_score(self) -> float:
        """Returns the quality-diversity score of the map."""
        return self.array[np.isfinite(self.array)].sum()

    @property
    def maximum(self) -> float:
        """Returns the maximum value in the map."""
        return self.array.max()

    @property
    def niches_filled(self) -> int:
        """Returns the number of niches in the map that have been explored."""
        return np.count_nonzero(np.isfinite(self.array))


class MAPElites:
    """
    Class implementing MAP-Elites, a quality-diversity algorithm.

    MAP-Elites creates a map of high perfoming solutions at each point in a
    discretized behavior space. First, the algorithm generates some initial random
    solutions, and evaluates them in the environment. Then, it  repeatedly mutates
    the solutions in the map, and places the mutated solutions in the map if they
    outperform the solutions already in their niche.
    """

    def __init__(
        self,
        env,
        map_grid_size: tuple[int, ...],
        init_map: Optional[Map] = None,
        history_length: int = 1,
        save_history: bool = False,
    ):
        """
        Class implementing MAP-Elites, a quality-diversity algorithm.

        Args:
            env (BaseEnvironment): The environment to evaluate solutions in. This
            should be a subclass of `BaseEnvironment`, and should implement
            methods to generate random solutions, mutate existing solutions,
            and evaluate solutions for their fitness in the environment.
            map_grid_size (int): Number of bins to partition the behavior space into.
            init_map (Map, optional): A map to use for the algorithm. If not passed,
            a new map will be created. Defaults to None.
            history_length (int): Length of history to store for each niche (cell)
            in the map. This acts as a circular buffer, so after storing
            `history_length` items, the buffer starts overwriting the oldest
            items.
            save_history (bool, optional): Whether to save the history of all
            generated solutions, even if they are not inserted into the map.
            Defaults to False.
        """
        self.env: BaseEnvironment = env
        self.map_grid_size = map_grid_size
        self.history_length = history_length
        self.save_history = save_history
        # self.history will be set/reset each time when calling `.search(...)`
        self.history: dict = defaultdict(list)
        # discretization of space
        # TODO: make this work for any number of dimensions
        self.bins = np.linspace(*env.behavior_space, map_grid_size[0] + 1)[1:-1].T  # type: ignore
        # TODO: abstract all maps out to a single class.
        # perfomance of niches
        if init_map is None:
            self.fitnesses: Map = Map(
                dims=map_grid_size * env.behavior_ndim,
                fill_value=-np.inf,
                dtype=float,
                history_length=history_length,
            )
        else:
            self.fitnesses = init_map
        # niches' sources
        self.genomes: Map = Map(
            dims=self.fitnesses.dims,
            fill_value=0.0,
            dtype=object,
            history_length=history_length,
        )
        # index over explored niches to select from
        self.nonzero: Map = Map(dims=self.fitnesses.dims, fill_value=False, dtype=bool)
        # bad mutations that ended up with invalid output.
        self.recycled = [None] * 1000
        self.recycled_count = 0

        print(f"MAP of size: {self.fitnesses.dims} = {self.fitnesses.map_size}")

    def to_mapindex(self, b: Phenotype) -> MapIndex:
        """Converts a phenotype (position in behaviour space) to a map index."""
        return (
            None
            if b is None
            else tuple(np.digitize(x, bins) for x, bins in zip(b, self.bins))
        )

    def random_selection(self) -> MapIndex:
        """Randomly select a niche (cell) in the map that has been explored."""
        ix = np.random.choice(np.flatnonzero(self.nonzero.array))
        return np.unravel_index(ix, self.nonzero.dims)

    def search(self, init_steps: int, total_steps: int, atol: float = 1.0) -> str:
        """
        Run the MAP-Elites search algorithm.

        Args:
            initsteps (int): Number of initial random solutions to generate.
            totalsteps (int): Total number of steps to run the algorithm for,
                including initial steps.
            atol (float, optional): Tolerance for how close the best performing
                solution has to be to the maximum possible fitness before the
                search stops early. Defaults to 1.

        Returns:
            str: A string representation of the best perfoming solution. The
                best performing solution object can be accessed via the
                `current_max_genome` class attribute.
        """
        tbar = trange(int(total_steps))
        max_fitness = -np.inf
        max_genome = None
        if self.save_history:
            self.history = defaultdict(list)

        for n_steps in tbar:
            if n_steps < init_steps or self.genomes.empty:
                # Initialise by generating initsteps random solutions.
                # If map is still empty: force to do generation instead of mutation.
                # TODO: use a separate sampler, move batch size to qd config.
                new_individuals: list[Genotype] = self.env.random()
            else:
                # Randomly select a batch of elites from the map.
                batch: list[Genotype] = []
                for _ in range(self.env.batch_size):
                    map_ix = self.random_selection()
                    batch.append(self.genomes[map_ix])
                # Mutate the elite.
                new_individuals = self.env.mutate(batch)

            # `new_individuals` is a list of generation/mutation. We put them
            # into the behavior space one-by-one.
            # TODO: account for the case where multiple new individuals are
            # placed in the same niche, for saving histories.
            for individual in new_individuals:
                print(individual)
                fitness = self.env.fitness(individual)
                if np.isinf(fitness):
                    continue
                map_ix = self.to_mapindex(individual.to_phenotype())
                # if the return is None, the individual is invalid and is thrown
                # into the recycle bin.
                if map_ix is None:
                    self.recycled[self.recycled_count % len(self.recycled)] = individual
                    self.recycled_count += 1
                    continue

                if self.save_history:
                    # TODO: thresholding
                    self.history[map_ix].append(individual)
                self.nonzero[map_ix] = True

                # If new fitness greater than old fitness in niche, replace.
                if fitness > self.fitnesses[map_ix]:
                    self.fitnesses[map_ix] = fitness
                    self.genomes[map_ix] = individual
                # If new fitness is the highest so far, update the tracker.
                if fitness > max_fitness:
                    max_fitness = fitness
                    max_genome = individual

                    tbar.set_description(f"{max_fitness=:.4f}")
                # Stop if best fitness is within atol of maximum possible fitness.
                if np.isclose(max_fitness, self.env.max_fitness, atol=atol):
                    break

        self.current_max_genome = max_genome
        return str(max_genome)

    def niches_filled(self):
        """Get the number of niches that have been explored in the map."""
        return self.fitnesses.niches_filled

    def maximum_fitness(self):
        """Get the maximum fitness value in the map."""
        return self.fitnesses.maximum

    def qd_score(self):
        """
        Get the quality-diversity score of the map.

        The quality-diversity score is the sum of the performance of all solutions
        in the map.
        """
        return self.fitnesses.qd_score
