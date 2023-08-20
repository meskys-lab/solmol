from pathlib import Path
from typing import Union

import pandas as pd


def parse_fasta(path: Union[Path, str]) -> pd.DataFrame:
    seq_data = pd.read_csv(path, lineterminator=">", header=None)
    seq_data = seq_data[["title", "sequence", "empty_col"]] = seq_data[0].str.split("\n", expand=True)
    seq_data = seq_data[["title", "sequence"]]
    seq_data["solubility"] = seq_data.title.apply(lambda x: x.split("|")[2]).astype(int)
    return seq_data
