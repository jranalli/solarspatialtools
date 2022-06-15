from setuptools import setup, find_packages

setup(
    name='solartoolbox',
    version='0.2.0',
    author="Joe Ranalli",
    author_email="jar339@psu.edu",
    description="A research toolbox for solar analysis",
    url="https://github.com/jranalli/solartoolbox",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas>=1.0',
        'tables',  # possibly must be installed separately via conda pytables because of MS dependency
        'numpy>=1.2',
        'pyproj>=3.0',
        'pvlib>=0.9',
        'netcdf4',
        'scipy>=1.6'
    ],
    license='BSD (3 Clause)',
    extras_require=dict(tests=['pytest'], demos=['matplotlib']),
)
