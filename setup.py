try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name = 'khutility',
    description = 'Utility Functions by and for Kyle',
    authon = 'Kyle Hershberger',
    version = '0.3.0',
    packages = ['khutility'],
    install_requires = [ ]
    )
