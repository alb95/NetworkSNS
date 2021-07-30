from setuptools import setup, find_packages

__author__ = 'Alberto Bucci'
__license__ = "BSD"
__email__ = "albertobucci1995@gmail.com"

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()

setup(name='networksns',
      version='0.0.1',
      license='BSD-Clause-2',
      description='Complex Network library',
      url='https://github.com/alb95/NetworkSNS',
      author='Alberto Bucci',
      author_email='albertobucci1995@gmail.com',
      use_2to3=True,
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 4 - Beta',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

          "Operating System :: OS Independent",

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python',
          'Programming Language :: Python :: 3'
      ],
      keywords='complex-networks',
      install_requires=['dynetx', 'numpy', 'networkx', 'scipy', 'statsmodels', ''],
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test", "networksns_test.py"]),
      )
