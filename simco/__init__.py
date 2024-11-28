# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import copy
import datetime
import multiprocessing as mp
import os
import random
import re
import string
from collections import Counter
from typing import Iterable, List, Optional

import abstar
import abutils
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm


def make_wells(num_wells: int) -> List[str]:
    """
    Generate a list of well names for a plate with a given number of wells.

    Parameters
    ----------
    num_wells : int
        The number of wells to generate. Must be between 1 and 384, inclusive.

    Returns
    -------
    List[str]
        A list of well names.

    Raises
    ------
    ValueError
        If `num_wells` is not between 1 and 384, inclusive.
    """
    if not 1 <= num_wells <= 384:
        raise ValueError("Number of wells must be between 1 and 384, inclusive.")
    num_cols = 12 if num_wells <= 96 else 24
    num_rows = 8 if num_wells <= 96 else 16
    columns = [f"{i:02}" for i in range(1, num_cols + 1)]
    rows = list(string.ascii_uppercase[:num_rows])
    well_list = [f"{r}{c}" for r in rows for c in columns]
    return well_list[:num_wells]
