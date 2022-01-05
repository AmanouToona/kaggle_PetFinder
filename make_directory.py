import pathlib


dir_list = [
    "input",
    "output",
    "log",
    "model_trained",
    "config",
    "models",
    "model_score",
    "utils",
]

for directory in dir_list:
    p = pathlib.Path(directory)

    if p.exists():
        continue

    p.mkdir()
