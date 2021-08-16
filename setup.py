import setuptools
from pathlib import Path
from setuptools.command.build_py import build_py
from subprocess import check_call,run
from distutils.core import Extension
import os
import shutil
import stat

# Borrowed from
# https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
# and modified. Thanks mertyldiran

class PreProcessing(build_py):
    """Pre-installation binary compilation."""
    def run(self):
        path = str((Path(__file__).parent).resolve())
        src_path = os.path.join(path,"rusty_axe","src")
        bin_dir_path = os.path.join(path,"rusty_axe","bin")
        bin_path =  os.path.join(bin_dir_path,"rf_5")
        compile_path = os.path.join(path,"target","release","rf_5")
        print(f"Building binary at {compile_path}")
        print(f"Placing binary at {bin_path}")
        try:
            os.mkdir(bin_dir_path)
        except FileExistsError:
            pass
        run(["cargo","build","--release"])
        os.replace(compile_path,bin_path)
        os.chmod(bin_path,stat.S_IRWXU)
        shutil.rmtree(src_path)
        build_py.run(self)


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="rusty-axe-bbrener1",
        version="0.6",
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
    )
