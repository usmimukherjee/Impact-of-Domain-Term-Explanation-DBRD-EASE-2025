# !pip install sentencepiece
# !pip install transformers
# !pip install torch
# !pip install rich

# Importing libraries

from rich import box
from rich.console import Console
# rich: for a better display on terminal
from rich.table import Column, Table

# define a rich console logger
console = Console(record=True)


# to display dataframe in ASCII format
def display_df(df):
    """display dataframe in ASCII format"""

    Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)


# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# Setting up the device for GPU usage
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
