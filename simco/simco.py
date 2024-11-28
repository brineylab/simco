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
from typing import Iterable, List, Optional, Union

import abstar
import abutils
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm


def create_well_list(num_wells: int) -> List[str]:
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


def simulate(
    parquet_file: str,
    output_directory: str,
    expansion_factor: Union[int, float] = 10,
    expansion_std: Optional[float] = None,
    num_wells: int = 96,
    cells_per_well: int = 5000,
    cells_per_well_std: Optional[float] = None,
    cell_dropout: Optional[float] = 0.3,
    reads_per_well: int = 3e5,
    reads_per_well_std: Optional[float] = None,
    reads_per_cell_std_multiplier: float = 0.25,
    light_chain_oversampling: float = 5.0,
    error_rate: Optional[float] = 1.0,
    error_std: Optional[float] = 0.5,
    # max_depth_of_sequencing=20,
):
    """
    Generates plates of sequencing data given a dataset of pairs with B cell expansion.
    Expanded pairs will have the same base name but are differentiated by version numbers.
    """
    # load data
    pairs = abutils.io.read_parquet(parquet_file)

    # metadata
    expansion_metadata = []
    well_metadata = []
    read_metadata = []

    # output directory
    if output_directory is not None:
        abutils.io.make_dir(output_directory)

    # expand each pair by expansion_factor and append version number (.1, .2, ...)
    #
    # if expansion_std is provided, randomly sample the expansion from a normal distribution
    # centered at expansion_factor and with deviation of expansion_std
    expanded_pairs = []
    print("expanding cells:")
    if expansion_std is not None:
        expansions = np.random.normal(
            loc=expansion_factor,
            scale=expansion_std,
            size=len(pairs),
        )
        expansions = np.round(expansions).astype(int)
        # expansions = [round(e) for e in expansions]
    else:
        expansions = [round(expansion_factor)] * len(pairs)
    for pair, expansion in tqdm(zip(pairs, expansions), total=len(pairs)):
        expanded = [copy.deepcopy(pair) for _ in range(expansion)]
        for i, p in enumerate(expanded, 1):
            p.name = f"{p.name}.{i}"
        expanded_pairs.extend(expanded)
        # log expansion metadata
        expansion_metadata.append(
            {
                "cell": pair.name,
                "expansion": expansion,
            }
        )

    # shuffle the expanded pairs
    print("\nshuffling expanded cells...")
    random.shuffle(expanded_pairs)

    # create wells
    wells = create_well_list(num_wells)
    well_dict = {}  # Create a dict for wells

    # distributed expanded cells into wells
    idx = 0
    print("\ndistributing expanded cells to wells...")
    if cells_per_well_std is not None:
        num_cells = [
            round(n)
            for n in np.random.normal(
                cells_per_well,
                cells_per_well_std,
                num_wells,
            )
        ]
    else:
        num_cells = [cells_per_well] * num_wells
    for well, n_cells in zip(wells, num_cells):
        well_dict[well] = expanded_pairs[idx : idx + n_cells]
        idx += n_cells

        # for cell in well_dict[well]:

    print(f"  - distributed {idx} cells to wells")

    # generate sequencing reads (optionally with errors)
    print("\ngenerating sequencing reads for each well:")
    if reads_per_well_std is not None:
        read_depths = np.random.normal(
            loc=reads_per_well, scale=reads_per_well * 0.25, size=len(well_dict)
        )
    else:
        read_depths = [reads_per_well] * len(well_dict)
    for read_depth, well_name in tqdm(
        zip(read_depths, well_dict.keys()), total=len(well_dict), leave=True
    ):
        well_cells = well_dict[well_name]
        reads = []
        if cell_dropout is not None:
            filtered_cells = []
            well_dropout = np.random.uniform(size=len(well_cells))
            for c, d in zip(well_cells, well_dropout):
                if d >= cell_dropout:
                    filtered_cells.append(c)
                else:
                    read_metadata.append(
                        {
                            "cell": c.name,
                            "well": well_name,
                            "dropout": True,
                        }
                    )
            well_cells = filtered_cells
            # well_cells = [c for c, d in zip(well_cells, well_dropout) if d >= cell_dropout]
        avg_reads_per_cell = read_depth / len(well_cells)
        avg_reads_per_heavy = avg_reads_per_cell / (light_chain_oversampling + 1)
        avg_reads_per_light = light_chain_oversampling * avg_reads_per_heavy
        heavy_depths = np.random.normal(
            avg_reads_per_heavy,
            scale=avg_reads_per_heavy * reads_per_cell_std_multiplier,
            size=len(well_cells),
        )
        heavy_depths = np.round(np.clip(heavy_depths, 0, None)).astype(int)
        light_depths = np.random.normal(
            avg_reads_per_light,
            scale=avg_reads_per_light * reads_per_cell_std_multiplier,
            size=len(well_cells),
        )
        light_depths = np.round(np.clip(light_depths, 0, None)).astype(int)
        for cell, heavy_depth, light_depth in tqdm(
            zip(well_cells, heavy_depths, light_depths),
            total=len(well_cells),
            leave=False,
            desc=well_name,
        ):
            # heavy chains
            h_seq = cell.heavy.sequence
            h_len = len(h_seq)
            h_avg_errors = error_rate / 100 * h_len
            h_std_errors = error_std / 100 * h_len
            h_num_errors = np.random.normal(
                h_avg_errors, h_avg_errors, size=round(heavy_depth)
            )
            h_num_errors = np.clip(h_num_errors, 0, None)
            h_num_errors = np.round(h_num_errors).astype(int)
            for i, h_err in enumerate(h_num_errors, 1):
                read = h_seq
                if h_err > 0:
                    positions = np.random.choice(h_len, size=h_err, replace=False)
                    for pos in positions:
                        original_base = h_seq[pos]
                        bases = ["A", "C", "G", "T"]
                        bases.remove(original_base)
                        base = np.random.choice(bases)
                        read = read[:pos] + base + read[pos + 1 :]
                reads.append(f">{cell.name}.{i}H\n{read}")

            # light chains
            l_seq = cell.heavy.sequence
            l_len = len(l_seq)
            l_avg_errors = error_rate / 100 * l_len
            l_std_errors = error_std / 100 * l_len
            l_num_errors = np.random.normal(
                l_avg_errors, l_avg_errors, size=round(light_depth)
            )
            l_num_errors = np.clip(l_num_errors, 0, None)
            l_num_errors = np.round(l_num_errors).astype(int)
            for i, l_err in enumerate(l_num_errors, 1):
                read = l_seq
                if l_err > 0:
                    positions = np.random.choice(l_len, size=l_err, replace=False)
                    for pos in positions:
                        original_base = l_seq[pos]
                        bases = ["A", "C", "G", "T"]
                        bases.remove(original_base)
                        base = np.random.choice(bases)
                        read = read[:pos] + base + read[pos + 1 :]
                reads.append(f">{cell.name}.{i}L\n{read}")

            read_metadata.append(
                {
                    "cell": cell.name,
                    "parent_cell": cell.name.split(".")[0],
                    "well": well_name,
                    "heavy_reads": heavy_depth,
                    "light_reads": light_depth,
                    "heavy_sequence": cell.heavy.sequence,
                    "light_sequence": cell.light.sequence,
                    "dropout": False,
                }
            )

        with open(f"./{output_directory}/{well_name}.fasta", "w") as f:
            f.write("\n".join(reads))

    expansion_df = pl.DataFrame(expansion_metadata)
    well_df = pl.DataFrame(well_metadata)
    read_df = pl.DataFrame(read_metadata, infer_schema_length=None)

    return expansion_df, well_df, read_df
