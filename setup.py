from setuptools import setup

from wordplay import __name__, __version__

setup(
    name=__name__,
    version=__version__,
    packages=[__name__],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        'Intended Audience :: Science/Research'],
    pyython_requires='>=3.6.8',
    install_requires=[
        'tabulate',
        'sortedcontainers',
        'pyprind',
        'seaborn',
        'scikit-learn',
        'scipy',
        'attr',
        'numpy',
        'matplotlib',
        'spacy',
        'pandas',
        'cytoolz',
        'pygal',
        'pingouin'
    ],
    url='https://github.com/phueb/Wordplay',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Analyze statistical properties of text related to lexical categories',

)