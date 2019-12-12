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
        'pingouin',
        'preppy @ git+https://github.com/phueb/Preppy.git#egg=Preppy-v1.3.0',
        'categoryeval @ git+https://github.com/phueb/CategoryEval.git#egg=CategoryEval-v1.1.0',
    ],
    url='https://github.com/phueb/Wordplay',
    license='',
    author='Philip Huebner',
    author_email='info@philhuebner.com',
    description='Analyze distributional properties of lexical categories in text',

)