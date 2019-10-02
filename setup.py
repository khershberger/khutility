try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name = 'khutility',
      description = 'Universal Utility Functions',
      authon = 'Kyle Hershberger',
      version = '0.1.1',
      packages = ['pyate'],
      install_requires = [
          'matplotlib',
		  'numpy',
          'pandas',
          'pyqt5',
          'scipy'
          ]
      )
