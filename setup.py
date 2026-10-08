from setuptools import find_packages, setup


setup(
    name="ldc-yolo",
    version="0.1.0",
    description="LDC-YOLO for aerial small-object detection",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
