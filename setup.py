import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rdp-quick",
    version="1.0.2",
    author="Dr John Dale",
    url="https://github.com/DrJohnDale/rdp-quick",
    description="fast implementation of Ramer-Douglas-Peucker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["rdp_quick"],
    license = "MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "numba", "scipy"]                     # Install other dependencies if any
)