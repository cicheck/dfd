import typing as t

import click

from dfd.datasets import preprocessor

from .dto import PreprocessDTO, pass_process_dto


# TODO: allow to copy files
@click.command(name="split")
@click.option("--train", type=click.FLOAT, default=None)
@click.option("--validation", type=click.FLOAT, default=None)
@click.option("--test", type=click.FLOAT, default=None)
@pass_process_dto
def split(
    preprocess_dto: PreprocessDTO,
    train: t.Optional[float],
    validation: t.Optional[float],
    test: t.Optional[float],
):
    if train is None and validation is None and test is None:
        train, validation, test = 0.6, 0.2, 0.2
    if train is None or validation is None or test is None:
        click.echo(
            "If at least one option from [train, validation, test]"
            + " is specified all must be specified."
        )
        return
    if train + validation + test != 1:
        click.echo("Ratios [train, validation, test] must adds to one.")
        return
    click.echo(f"Splitting with ratios [train: {train}, validation: {validation}, test: {test}].")
    preprocessor.split(
        input_path=preprocess_dto.input_path,
        output_path=preprocess_dto.output_path,
        train_ds_share=train,
        validation_ds_share=validation,
        test_ds_share=test,
    )
