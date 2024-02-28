from setuptools import find_packages, setup

setup(
        name='alg',
        version='0.0.1',
        packages=find_packages(include=['src']),
        package_dir={'':'src'}
        )
