import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="siamese_rit_change",
    version="0.0.1",
    author="Bhavan Vasu, Faiz Ur Rahman, Andreas Savakis",
    author_email="bxv7657@rit.edu",
    description="A package code for aerial patch based change detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vbhavank/Siamese-neural-network-for-change-detection",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ),
)
