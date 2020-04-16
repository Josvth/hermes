from setuptools import setup, find_packages

setup(
    name = 'hermes-simulator',
    packages = find_packages(),
    version = 'v0.1.3beta',
    license = 'GPL3+',
    description = 'Simulator for satellite data relay systems',
    author = 'Jos van \'t Hof',
    author_email = 'josvth@gmail.com',
    url = 'https://github.com/Josvth/hermes-simulator',
    download_url = 'https://github.com/Josvth/hermes-simulator/archive/v0.1.2beta.tar.gz',
    keywords = ['data', 'relay', 'satellite'],
    install_requires=['poliastro',
                      'PyQt5',
                      'mayavi'],
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
