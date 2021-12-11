import contextlib
import io

from pytest import mark

from dfd.cli.utils import echo_metrics


@mark.parametrize(
    "given_metrics, expected_string",
    [
        (
            {"foo": 0, "bar": 0},
            (
                "| metrics  | values   |\n"
                + "|:---:     |:---:     |\n"
                + "|foo       |0.00000   |\n"
                + "|bar       |0.00000   |\n"
                + "\n"
            ),
        ),
        (
            {},
            ("| metrics  | values   |\n" + "|:---:     |:---:     |\n" + "\n"),
        ),
    ],
)
def test_echo_metrics(given_metrics, expected_string):
    with contextlib.redirect_stdout(io.StringIO()) as captured_stdout:
        echo_metrics(given_metrics, width=10)
        assert captured_stdout.getvalue() == expected_string
