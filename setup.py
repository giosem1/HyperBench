from setuptools import find_packages, setup

setup(
    name='hypernegative',
    version='0.1.7',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pipeline=pipelines.pipeline:execute',
        ],
    },
    install_requires=[],
)