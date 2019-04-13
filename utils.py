import os


def create_path(*parts):
    # path = os.path.dirname(__file__)
    path = os.getcwd()
    for part in parts:
        path = os.path.join(path, part)
    return path