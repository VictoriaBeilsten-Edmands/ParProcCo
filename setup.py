from setuptools import setup

setup(
    name='ParProcCo',
    version='1.0',
    description='Parallel Processing Coordinator. Splits dataset processing to run parallel clister jobs and aggregates outputs',
    packages=['ParProcCo'],
    install_requires=['h5py', 'numpy'],
)
