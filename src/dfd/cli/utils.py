import typing as t

import click


def _build_single_row(*columns, width: int) -> str:
    row_as_string = "|"
    for column in columns:
        row_as_string += column.ljust(width, " ")
        row_as_string += "|"
    row_as_string += "\n"
    return row_as_string


def echo_metrics(metrics_map: t.Dict[str, float], width: int = 20):
    """Echo given metrics map as markdown table.

    Args:
        metrics_map: Name of metrics mapped to their values.
        width: Min width of table cell

    """
    first_column_name = " metrics"
    second_column_name = " values"
    table_as_string = ""
    table_as_string += _build_single_row(first_column_name, second_column_name, width=width)
    header_sign = ":---:"
    table_as_string += _build_single_row(header_sign, header_sign, width=width)
    for metric, value in metrics_map.items():
        table_as_string += _build_single_row(metric, "{:.5f}".format(value), width=width)
    click.echo(table_as_string)
