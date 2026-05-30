from setuptools import find_packages, setup


setup(
    name="mkl-yolo",
    version="0.1.0",
    description="MKL-YOLO for aerial small-object detection",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
