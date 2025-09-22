from setuptools import find_packages, setup

setup(
    name='hypernegative',
    version='0.1.2',
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'imdb_pipeline=pipelines.imdb_pipeline:execute',
            'arb_pipeline=pipelines.arb_pipeline:execute'
        ],
    },
    install_requires=[],
)