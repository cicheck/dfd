import typing as t

import click

from dfd.datasets import preprocessor

from .dto import PreprocessDTO, pass_process_dto


# TODO: allow to copy files
@click.command(name="split")
@click.option("train_share", "--train", type=click.FLOAT, default=None)
@click.option("validation_share", "--validation", type=click.FLOAT, default=None)
@click.option("test_share", "--test", type=click.FLOAT, default=None)
@pass_process_dto
def split(
    preprocess_dto: PreprocessDTO,
    train_share: t.Optional[float],
    validation_share: t.Optional[float],
    test_share: t.Optional[float],
):
    """Split given dataset into train, validation and test datasets.

    Args:
        preprocess_dto: Object containing input and output path, passed via decorator.
        train_share: Share of training dataset.
        validation_share: Share of validation dataset.
        test_share: Share of test dataset.

    """
    if train_share is None and validation_share is None and test_share is None:
        train_share, validation_share, test_share = 0.6, 0.2, 0.2
    if train_share is None or validation_share is None or test_share is None:
        click.echo(
            "If at least one option from [train, validation, test]"
            + " is specified all must be specified."
        )
        return
    if train_share + validation_share + test_share != 1:
        click.echo("Ratios [train, validation, test] must adds to one.")
        return
    click.echo(f"Splitting with ratios [train: {train_share}, validation: {validation_share}, test: {test_share}].")
    preprocessor.split(
        input_path=preprocess_dto.input_path,
        output_path=preprocess_dto.output_path,
        train_ds_share=train_share,
        validation_ds_share=validation_share,
        test_ds_share=test_share,
    )
