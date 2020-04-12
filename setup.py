from setuptools import setup, find_packages

setup(
    name = 'hermes-simulator',
    packages = find_packages(),
    version = 'v0.1-beta',
    license = 'GPL3+',
    description = 'Simulator for satellite data relay systems',
    author = 'Jos van \'t Hof',
    author_email = 'josvth@gmail.com',
    url = 'https://github.com/Josvth/hermes-simulator',
    download_url = 'https://github.com/Josvth/hermes-simulator/archive/0.1.tar.gz',
    keywords = ['data', 'relay', 'satellite'],
    install_requires=['numpy',
                      'poliastro',
                      'scipy',
                      'astropy',
                      'PyQt5',
                      'mayavi',
                      'numba'],
    classifiers = [
                   'Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',  # Define that your audience are developers
                   'Topic :: Software Development :: Build Tools',
                   'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',  # Again, pick a license
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6',
               ],
)
