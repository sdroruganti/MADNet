from setuptools import find_packages, setup

setup(
    name="madnet",
    version="0.1.0",
    description=(
        "A multimodal dense correspondence model for cross-modal image alignment"
    ),
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=2.5",
        "Pillow>=12.2.0",
        "PyYAML>=6.0",
        "torch>=2.12",
        "torchvision>=0.27",
        "opencv-python>=4.13",
    ],
)

