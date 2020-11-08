import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()


setup(name = 'pnumpy',
      version = '1.4.2',
      description = 'A very lightweight implementation of distributed arrays',
      long_description = README,
      long_description_content_type="text/markdown",
      author = 'Alexander Pletzer',
      author_email = 'alexander@gokliya.net',
      url = 'https://github.com/pletzer/pnumpy.git',
      install_requires=['numpy', 'mpi4py'],
      package_dir={'pnumpy': 'src'}, # the present directory maps to src 
      packages = ['pnumpy'],
)
