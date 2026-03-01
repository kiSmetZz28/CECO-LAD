import random
from itertools import product
from typing import Dict, Tuple

# predefined parameter lists
num_epochs_list = [3, 6, 10]
k_list = [3, 4, 5]
e_layer_num_list = [3, 6, 8]
batch_size_list = [32, 64, 96]

# generate all combinations
all_combinations = list(product(num_epochs_list, k_list, e_layer_num_list, batch_size_list))

# generate unique random state values
random.seed(42)
unique_random_states = random.sample(range(10, 301), len(all_combinations))

# mapping from combination to random state
combo_to_random_state: Dict[Tuple[int, int, int, int], int] = {
    combo: state for combo, state in zip(all_combinations, unique_random_states)
}


def get_random_state(num_epochs: int, k: int, e_layer_num: int, batch_size: int) -> int:
    """
    return a unique random state for the given parameter combination

    args:
        num_epochs (int): one of [3, 6, 10]
        k (int): one of [3, 4, 5]
        e_layer_num (int): one of [3, 6, 9]
        batch_size (int): one of [32, 64, 96]

    returns:
        int: unique random state in range [10, 200]
    """
    key = (num_epochs, k, e_layer_num, batch_size)
    if key not in combo_to_random_state:
        raise ValueError(f"invalid combination: {key}")

    print(f"get combination {key}, the random state is {combo_to_random_state[key]}")
    return combo_to_random_state[key]
