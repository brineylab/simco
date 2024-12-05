# Copyright (c) 2024 Bryan Briney
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


import os
import random
import string
from typing import Iterable, List, Optional, Union

import abutils
import numpy as np
import polars as pl
from tqdm.auto import tqdm


def simulate(
    project_directory: str,
    parquet_file: Optional[str] = None,
    num_input_cells: Optional[int] = None,
    expansion_factor: Union[int, float, Iterable[Union[int, float]]] = 10,
    expansion_std: Optional[float] = None,
    num_wells: int = 96,
    cells_per_well: int = 5000,
    cells_per_well_std: Optional[float] = None,
    cell_dropout: float = 0.0,
    reads_per_well: int = 3e5,
    reads_per_well_std: Optional[float] = None,
    reads_per_cell_std_multiplier: float = 0.25,
    light_chain_oversampling: float = 5.0,
    error_rate: Optional[float] = 1.0,
    error_std: Optional[float] = 0.5,
    verbose: bool = True,
):
    """
    Generates synthetic co-occurence sequencing data.

    Parameters
    ----------
    project_directory : str
        Path to a directory where the output and logs will be saved.

    parquet_file : str
        Path to a parquet file containing a table of pairs.

    num_input_cells : int or None
        The number of input cells to simulate. If not provided, the number of input cells will be
        inferred from `num_wells` and `cells_per_well`.

        .. note::
            The built-in dataset contains ~1.4M input cells. If using this dataset, values for
            `num_input_cells` greater than the size of the built-in dataset will be ignored and
            the size of the built-in dataset will be used instead.

    expansion_factor : int, float, or Iterable[int, float], default=10
        The average number of times each pair will be expanded. If a `expansion_factor` is a
        `float` and `expansion_std` is not provided, `expansion_factor` will be rounded to the
        nearest integer. If `expansion_factor` is an `Iterable`, it must be the same length as
        `pairs`.

    expansion_std : float or None
        The standard deviation of the number of times each pair will be expanded. If provided,
        the number of times each input pair is expanded will be sampled from a normal distribution
        centered at `expansion_factor` and with a standard deviation of `expansion_std`. If not
        provided, each input pair will be expanded exactly `expansion_factor` times.

    num_wells : int, default=96
        The number of wells to simulate. Must be between 1 and 384, inclusive.

    cells_per_well : int, default=5000
        The average number of cells per well.

    cells_per_well_std : float or None
        The standard deviation of the number of cells per well. If provided, the number of cells
        per well will be sampled from a normal distribution centered at `cells_per_well` and with
        a standard deviation of `cells_per_well_std`. If not provided, the number of cells per
        well will be exactly `cells_per_well`.

    cell_dropout : float, default=0.0
        The fraction of cells to drop from each well.

    reads_per_well : int, default=3e5
        The average number of reads per well.

    reads_per_well_std : float or None
        The standard deviation of the number of reads per well. If provided, the number of reads
        per well will be sampled from a normal distribution centered at `reads_per_well` and with
        a standard deviation of `reads_per_well_std`. If not provided, the number of reads per
        well will be exactly `reads_per_well`.

    reads_per_cell_std_multiplier : float, default=0.25
        The standard deviation multiplier for the number of reads per cell.

    light_chain_oversampling : float, default=5.0
        The oversampling factor for the number of light chains per heavy chain.

    error_rate : float or None, default=1.0
        The sequencing error rate (in percent) for the heavy and light chains. For example,
        `1.0` corresponds to 1% error rate.

    error_std : float or None, default=0.5
        The standard deviation of the error rate for the heavy and light chains. If provided,
        the error rate for the heavy and light chains will be sampled from a normal distribution
        centered at `error_rate` and with a standard deviation of `error_std`. If not provided,
        the error rate for the heavy and light chains will be exactly `error_rate`.

    verbose : bool, default=True
        Whether to print progress messages to the terminal.

    Raises
    ------
    ValueError
        If `expansion_factor` is an `Iterable` and its length does not match the number of pairs.
    """
    # load data
    if verbose:
        print("loading input data...")
    if parquet_file is None:
        # use built-in dataset if no parquet file is provided
        parquet_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/simco.parquet"
        )
    if num_input_cells is None:
        # use a rough estimate of the required number of input cells if not provided
        if isinstance(expansion_factor, Iterable):
            mean_expansion = np.mean(expansion_factor)
        else:
            mean_expansion = expansion_factor
        num_input_cells = round(1.25 * num_wells * cells_per_well / mean_expansion)
    num_input_cells = min(num_input_cells, 1408808)
    pairs = abutils.io.read_parquet(parquet_file)
    pairs = [p for p in pairs if p.is_pair][:num_input_cells]
    if verbose:
        print(f"  -> loaded {len(pairs)} cells")

    # metadata
    expansion_metadata = []
    read_metadata = []

    # project directories
    project_directory = os.path.abspath(project_directory)
    abutils.io.make_dir(project_directory)
    log_directory = os.path.join(project_directory, "logs")
    abutils.io.make_dir(log_directory)
    output_directory = os.path.join(project_directory, "fastas")
    abutils.io.make_dir(output_directory)

    # expansion
    expanded_pairs = []
    if verbose:
        print("")
        print("expanding cells:")
    if expansion_std is not None:
        expansions = np.random.normal(
            loc=expansion_factor,
            scale=expansion_std,
            size=len(pairs),
        )
        expansions = np.round(np.clip(expansions, 1, None)).astype(int)
    elif isinstance(expansion_factor, Iterable):
        if len(expansion_factor) != len(pairs):
            raise ValueError(
                "If expansion_factor is an iterable, it must be the same length as num_input_cells."
            )
        expansions = [round(e) for e in expansion_factor]
    else:
        expansions = [round(expansion_factor)] * len(pairs)
    for pair, expansion in tqdm(
        zip(pairs, expansions), total=len(pairs), disable=not verbose
    ):
        for exp in range(expansion):
            p = abutils.Pair(
                name=f"{pair.name}.{exp + 1}",
                sequences=[],
            )
            p.heavy = abutils.Sequence(pair.heavy.sequence, id=pair.heavy.id)
            p.light = abutils.Sequence(pair.light.sequence, id=pair.light.id)
            expanded_pairs.append(p)
        expansion_metadata.append(
            {
                "cell": pair.name,
                "expansion": expansion,
            }
        )

    # shuffle the expanded pairs
    if verbose:
        print("")
        print("shuffling expanded cells...")
    random.shuffle(expanded_pairs)

    # create wells
    wells = create_well_list(num_wells)
    well_dict = {}  # Create a dict for wells

    # distribute expanded cells into wells
    if verbose:
        print("")
        print("distributing expanded cells to wells...")
    idx = 0
    if cells_per_well_std is not None:
        num_cells = np.random.normal(
            cells_per_well,
            cells_per_well_std,
            size=num_wells,
        )
        num_cells = np.round(np.clip(num_cells, 0, None)).astype(int)
    else:
        num_cells = [cells_per_well] * num_wells
    for well, n_cells in zip(wells, num_cells):
        well_dict[well] = expanded_pairs[idx : idx + n_cells]
        idx += n_cells
    if verbose:
        print(f"  -> distributed {idx} cells to wells")

    # generate sequencing reads
    if verbose:
        print("")
        print("generating sequencing reads for each well:")
    # compute how many reads to generate for each well
    if reads_per_well_std is not None:
        read_depths = np.random.normal(
            loc=reads_per_well, scale=reads_per_well * 0.25, size=len(well_dict)
        )
        read_depths = np.round(np.clip(read_depths, 0, None)).astype(int)
    else:
        read_depths = [reads_per_well] * len(well_dict)
    for read_depth, well_name in tqdm(
        zip(read_depths, well_dict.keys()), total=len(well_dict), leave=True
    ):
        well_cells = well_dict[well_name]
        reads = []
        if cell_dropout > 0.0:
            filtered_cells = []
            well_dropout = np.random.uniform(size=len(well_cells))
            for c, d in zip(well_cells, well_dropout):
                if d >= cell_dropout:
                    filtered_cells.append(c)
                else:
                    read_metadata.append(
                        {
                            "cell": c.name,
                            "parent_cell": c.name.split(".")[0],
                            "well": well_name,
                            "heavy_reads": 0,
                            "light_reads": 0,
                            "heavy_sequence": c.heavy.sequence,
                            "light_sequence": c.light.sequence,
                            "dropout": True,
                        }
                    )
            well_cells = filtered_cells
        # divide reads per well across cells
        avg_reads_per_cell = read_depth / len(well_cells)
        heavy_depths = avg_reads_per_cell / (light_chain_oversampling + 1)
        light_depths = light_chain_oversampling * heavy_depths
        if reads_per_cell_std_multiplier is not None:
            heavy_depths = np.random.normal(
                heavy_depths,
                scale=heavy_depths * reads_per_cell_std_multiplier,
                size=len(well_cells),
            )

            light_depths = np.random.normal(
                light_depths,
                scale=light_depths * reads_per_cell_std_multiplier,
                size=len(well_cells),
            )
        heavy_depths = np.round(np.clip(heavy_depths, 0, None)).astype(int)
        light_depths = np.round(np.clip(light_depths, 0, None)).astype(int)

        # generate reads for each cell
        for cell, heavy_depth, light_depth in tqdm(
            zip(well_cells, heavy_depths, light_depths),
            total=len(well_cells),
            leave=False,
            desc=well_name,
        ):
            # heavy chains
            h_seq = cell.heavy.sequence
            h_len = len(h_seq)
            h_num_errors = error_rate / 100 * h_len
            if error_std is not None:
                h_std_errors = error_std / 100 * h_len
                h_num_errors = np.random.normal(
                    h_num_errors, h_std_errors, size=heavy_depth
                )
            else:
                h_num_errors = np.full(heavy_depth, h_num_errors)
            h_num_errors = np.round(np.clip(h_num_errors, 0, None)).astype(int)
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
            l_seq = cell.light.sequence
            l_len = len(l_seq)
            l_num_errors = error_rate / 100 * l_len
            if error_std is not None:
                l_std_errors = error_std / 100 * l_len
                l_num_errors = np.random.normal(
                    l_num_errors, l_std_errors, size=light_depth
                )
            else:
                l_num_errors = np.full(light_depth, l_num_errors)
            l_num_errors = np.round(np.clip(l_num_errors, 0, None)).astype(int)
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

        with open(f"{output_directory}/{well_name}.fasta", "w") as f:
            f.write("\n".join(reads))

    # log metadata
    expansion_df = pl.DataFrame(expansion_metadata)
    expansion_df.write_csv(os.path.join(log_directory, "expansion_metadata.csv"))
    read_df = pl.DataFrame(read_metadata, infer_schema_length=None)
    read_df.write_csv(os.path.join(log_directory, "read_metadata.csv"))


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
