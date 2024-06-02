from setuptools import find_packages, setup


def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


setup(
    name="trajectory_image_dataset",
    version="0.0.1",
    packages=find_packages("src"),
    author="Marcin Kostrzewa",
    author_email="Manik24901@gmail.com",
    description="A package for CPD and classfication of anomalous diffusion trajectories.",
    long_description=read_file("README.md"),
)
