.. hermes-simulator

:Name: Hermes Simulator
:Author: Jos van 't Hof
:Version: 0.1.3beta

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
   :target: https://docs.poliastro.space/en/latest/?badge=latest

.. |license| image:: https://img.shields.io/github/license/josvth/hermes-simulator
   :target: https://github.com/josvth/hermes-simulator/raw/master/LICENSE
   
.. |astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/

|docs| |license| |astropy|

Installation
============

Improved install (Windows)
--------------------------

1. Install "Build tools for Visual studio"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Go to the `Visual studio download page <https://visualstudio.microsoft.com/downloads/>`_
2. Scroll down and open the "Tools for Visual Studio 2019" drop down
3. Download "Build Tools for Visual Studio 2019"
4. Open executable and follow installation instructions
5. Make sure to install the  'Windows 10 SDK', 'Visual C++ tools for CMake' and 'MSVC'
6. Restart computer

2. Install hermes-simulator from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
1. Clone the git repository

.. code-block:: bash

    git clone https://github.com/Josvth/hermes-simulator.git

2. While in the cloned directory type the following command

.. code-block:: bash

    pip install hermes-simulator

.. note::   If you intend to modify/improve hermes-simulator I recommend installing using the '-e' (development mode) option

Original install
----------------

Installing Hermes is a bit tedious at the moment but the following squence of commands should make it all work.

Open the Anaconda terminal and make a new conda enviroment using:

.. code-block:: bash

   conda create --name hermes_environment python=3.7
   
And activate it using:

.. code-block:: bash

   conda activate hermes_environment
   
Then install as followed:

.. code-block:: bash

   conda install vtk
   conda install traits
   pip install hermes-simulator

Examples
============

Then start python using:

.. code-block:: bash

    python

And run the example by doing:

.. code-block:: python

    from hermes.examples import O3b_example

    O3b_example()
