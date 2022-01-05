from setuptools import setup, find_packages

setup(
    name='torsk',
    description='Anomaly Detection in Chaotic Time Series based on an ESN',
    author='Niklas Heim',
    author_email='heim.niklas@gmail.com',
    packages=find_packages(),
    version=0.1,
    install_requires=[
        "pandas",
        "scipy",
        "joblib",
        "marshmallow",
        "matplotlib",
        "netCDF4",
        "numpy",
    ]
)
