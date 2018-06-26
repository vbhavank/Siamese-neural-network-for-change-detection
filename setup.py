import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="siamese_change",
    version="0.0.1",
    author="Bhavan Vasu, Faiz Ur Rahman, Andreas Savakis",
    author_email="bxv7657@rit.edu",
    description="A package code for aerial patch based change detection",
    long_description=long_description,
    long_description_content_type="We present a patch-based algorithm for detecting structural changes in satellite imagery using a Siamese neural network. The two channels of our Siamese network are based on the VGG16 architecture with shared weights. Changes between the target and reference images are detected with a fully connected decision network that was trained on DIRSIG simulated samples and achieved a high detection rate. Alternatively, a change detection approach based on Euclidean distance between deep convolutional features achieved very good results with minimal supervision.",
    url="https://github.com/vbhavank/Siamese-neural-network-for-change-detection",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
    ),
)
