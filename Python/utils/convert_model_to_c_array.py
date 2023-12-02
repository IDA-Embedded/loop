"""
Convert TfLite model to C array
================================


Convert a tflite model into C array to be loaded into the intepreter at runtime on embedded targets.


"""


import rich_click as click
from tensorflow.lite.python import util

click.rich_click.USE_MARKDOWN = True


def insert_copyright_notice(source_path: str, header_path: str):
    """Insert Copyright into header and source file"""

    copyright = """/**
This is a dummy copyright header! 2023!
*/\n\n\n\n\n"""
    for i in range(2):
        path = [lambda: source_path, lambda: header_path][i == 1]()
        seach_term = "#"
        with open(path, "r+") as file:
            contents = file.read()
            index = contents.find(seach_term)
            file.seek(0)
            file.truncate()
            file.write(f"{copyright}{contents[index:]}")


def convert_tflite_to_c(settings: dict):
    """
    Convert tflite model to c array using tensorflow api
    """
    #   Creating paths to store the two files
    header_path = f'{settings["save_path"]}/{settings["file_name"]}.h'
    source_path = f'{settings["save_path"]}/{settings["file_name"]}.c'

    with open(settings["model_path"], "rb") as input_handle:
        input_data = input_handle.read()

    source, header = util.convert_bytes_to_c_source(
        data=input_data,
        array_name=settings["array_name"],
        max_line_width=80,
        include_guard=settings["file_name"],
        include_path=f'{settings["file_name"]}.h',
        use_tensorflow_license=False,
    )

    with open(source_path, "w") as source_handle:
        source_handle.write(source)

    with open(header_path, "w") as header_handle:
        header_handle.write(header)

    insert_copyright_notice(source_path, header_path)


@click.command()
@click.option("--model", required=True, type=str, help="Path to tflite model")
@click.option(
    "--array-name", required=True, type=str, help="name of the generated c-array"
)
@click.option(
    "--path-to-save",
    required=True,
    type=click.Path(exists=True),
    help="Path to store the generated header and source file",
)
@click.option(
    "--file-names", type=str, required=True, help="Name of the header and source file"
)
def cli(model: str, array_name: str, path_to_save, file_names: str):
    """Convert .tflite file into C source code"""

    settings = {
        "model_path": model,
        "array_name": array_name,
        "save_path": path_to_save,
        "file_name": file_names,
    }

    convert_tflite_to_c(settings)


if __name__ == "__main__":
    cli()
