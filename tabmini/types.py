from typing import Generator, Tuple
import pandas as pd

# Type alias for the dataset
# This is a generator that yields tuples of the dataset name and the dataset itself
# The dataset itself is a tuple of the features and the target
TabminiDataset = Generator[Tuple[str, Tuple[pd.DataFrame, pd.DataFrame]], None, None]
