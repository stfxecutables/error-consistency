from __future__ import annotations
import numpy as np

from numpy.random import SeedSequence, Generator
from numpy import ndarray


from typing import List, Optional, Union


MAX_SEED = 2 ** 32 - 1


def parallel_seed_generators(
    seed: Optional[int], repetitions: int, return_generators: bool = True
) -> Union[List[int], List[Generator]]:
    """Yield streams for generating random numbers in different processes that won't collide / be
    badly correlated

    Notes
    -----
    We *may* not need this, but it is better to avoid these very subtle potential gotchas
    discussed in the `NumPy guide <https://numpy.org/doc/stable/reference/random/parallel.html>`,
    this `blog post <https://albertcthomas.github.io/good-practices-random-number-generators/>`, and
    this `issue <https://github.com/numpy/numpy/issues/9650>`

    :meta private:
    """

    # attempt 4 fails because sklearn requires legacy generation
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    ss = rng.bit_generator._seed_seq
    seed_seqs: List[SeedSequence] = ss.spawn(repetitions)
    generators = [np.random.default_rng(s) for s in seed_seqs]
    if return_generators:
        return generators
    else:
        return [int(generator.integers(0, MAX_SEED, 1)) for generator in generators]


def random_seeds(seed: Optional[int], repetitions: int) -> ndarray:
    """Naively grab random seeds.

    :meta private:
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    return rng.integers(0, MAX_SEED, repetitions)
