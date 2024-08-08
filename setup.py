from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

ver = '0.4.1'


setup(
    name='solartoolbox',
    version=ver,
    author="Joe Ranalli",
    author_email="jranalli@psu.edu",
    description="A research toolbox for solar analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jranalli/solartoolbox",
    download_url="https://github.com/jranalli/solartoolbox/archive/{}.tar.gz".format(ver),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy<2',
        'pandas',
        'tables',  # possibly must be installed separately via conda pytables because of MS dependency
        'pyproj',
        'pvlib',
        'netcdf4',
        'scipy'
    ],
    license='BSD (3 Clause)',
    extras_require=dict(tests=['pytest'], demos=['matplotlib', 'jupyter'], docs=['sphinx', 'nbsphinx', 'sphinx_rtd_theme']),
)
