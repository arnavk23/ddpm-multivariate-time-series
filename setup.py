from setuptools import setup, find_packages

setup(
    name='ddpm-multivariate-time-series',
    version='0.1.0',
    author='Arnav K.',
    author_email='your_email@example.com',
    description='A project for training and evaluating DDPM on multivariate time series data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/arnavk23/ddpm-multivariate-time-series',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'torch',  # or any other deep learning library you are using
        'scikit-learn',  # if needed
        # add other dependencies as required
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)