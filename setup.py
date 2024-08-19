from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

ver = '0.4.5'


setup(
    name='solarspatialtools',
    version=ver,
    author="Joe Ranalli",
    author_email="jranalli@psu.edu",
    description="Spatial analysis tools for solar energy research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jranalli/solarspatialtools",
    download_url="https://github.com/jranalli/solarspatialtools/archive/{}.tar.gz".format(ver),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'numpy<2',
        'pandas<=2.2.2',
        'tables<=3.9.2',
        'pyproj<=3.6.1',
        'pvlib<=0.11.0',
        'netcdf4<1.7.2',
        'scipy<=1.14.0',
        'matplotlib<=3.9.2',
    ],
    license='BSD (3 Clause)',
    extras_require=dict(tests=['pytest'], demos=['jupyter'], docs=['sphinx', 'nbsphinx', 'sphinx_rtd_theme', 'ipython']),
)
