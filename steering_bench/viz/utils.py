import pandas as pd
from IPython.display import display, HTML


def display_df(df: pd.DataFrame) -> None:
    """Display a dataframe in an IPython notebook"""
    display(df)


def make_pretty_print_html(df: pd.DataFrame) -> HTML:
    """Make the HTML to pretty print a dataframe in an IPython notebook"""
    return HTML(
        df.to_html().replace("\\n", "<br>")  # Visualize newlines nicely
        # TODO: Figure out how to align text left
    )
