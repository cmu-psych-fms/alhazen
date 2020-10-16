from setuptools import setup
from alhazen import __version__

DESCRIPTION = open("README.md").read()

setup(name="alhazen",
      version=__version__,
      description="A simple framework to facilitate running experiments, written in Python, using cognitive models, or similar applications, in multiple, parallel processes",
      author="Don Morrison",
      author_email="dfm2@cmu.edu",
      url="https://bitbucket.org/dfmorrison/alhazen/",
      platforms=["any"],
      long_description=DESCRIPTION,
      long_description_content_type="text/markdown",
      py_modules=["alhazen"],
      install_requires=["tqdm"],
      tests_require=["pytest"],
      python_requires=">=3.7",
      classifiers=["Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Operating System :: OS Independent"])
