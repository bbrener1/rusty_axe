import setuptools
from pathlib import Path
from setuptools.command.build_py import build_py
from subprocess import check_call,run
from distutils.core import Extension

# Borrowed from
# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
# and modified. Thanks mertyldiran

class PreProcessing(build_py):
    """Pre-installation binary compilation."""
    def run(self):
        path = str((Path(__file__).parent).resolve())
        print(f"Building binary at {path}")
        run(["bash", "./recipe/py_build.sh",path],check=True)
        # run('ls'.split(),check=True)
        # raise Exception()
        build_py.run(self)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="rusty-axe-bbrener1",
        version="0.5",
        author="Boris Brenerman",
        author_email="bbrener1@jhu.edu",
        description="Random Forest Latent Structure (Biology)",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/bbrener1/rf_5",
        project_urls={
            "Bug Tracker": "https://github.com/bbrener1/rf_5/issues",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX",
            "Programming Language :: Rust",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
        packages=['rusty_axe'],
        package_dir={
            "rusty_axe": "./rusty_axe",
        },
        include_package_data=True,
        package_data={
            'figures':["figures/*.ipynb",],
            'bin':["bin/*",]
        },
        cmdclass={
            'build_py' : PreProcessing,
        },
        install_requires=[
            'scanpy',
            'leidenalg',
            'scikit-learn',
            'matplotlib>=3.4.2'
        ],
        python_requires="<3.9",
    )
