from distutils.core import setup

setup(
    name = 'hermes-simulator',
    packages = ['src/hermes'],
    version = '0.1',
    license = 'GPLv3+',
    description = 'Simulator for satellite data relay systems',
    author = 'Jos van \'t Hof',
    author_email = 'josvth@gmail.com',
    url = 'https://github.com/Josvth/hermes',
    download_url = 'https://github.com/Josvth/hermes/archive/0.1.tar.gz',
    keywords = ['data', 'relay', 'satellite'],
    install_requires=['numpy',
                      'poliastro',
                      'scipy',
                      'astropy',
                      'mayavi',
                      'PyQt5',
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
