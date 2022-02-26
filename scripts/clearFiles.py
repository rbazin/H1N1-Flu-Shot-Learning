import os


def clear_files(path):
    for (root, _, files) in os.walk(path):
        for file in files:
            print(file)
            os.remove(os.path.join(root, file))


if __name__ == '__main__':

    PATH_SCRIPTS = os.getcwd()
    PATH_MODELS = os.path.join(PATH_SCRIPTS, "..", "models")
    clear_files(os.path.join(PATH_MODELS, "chain1"))
    clear_files(os.path.join(PATH_MODELS, "chain2"))
