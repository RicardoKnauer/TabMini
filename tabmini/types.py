from typing import Tuple

import pandas as pd

# Type alias for the dataset
# This is a dict of the dataset name and the dataset itself
# The dataset itself is a tuple of the features and the target
TabminiDataset = dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
