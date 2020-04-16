.. hermes-simulator

:Name: Hermes Simulator
:Author: Jos van 't Hof
:Version: 0.1.3beta

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat-square
   :target: https://docs.poliastro.space/en/latest/?badge=latest

.. |license| image:: https://img.shields.io/github/license/josvth/hermes-simulator   :alt: GitHub

.. |astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat-square
   :target: http://www.astropy.org/

|docs| |license| |astropy|

< Some description >

Installation
============

Installing Hermes is a bit tedious at the moment but the following squence of commands should make it all work.

First make a new conda enviroment using:

.. code-block:: bash

   conda create --name hermes_environment python=3.7
   
And activate it using

.. code-block:: bash

   conda activate hermes_environment
   
Then install as followed:

.. code-block:: bash

   conda install vtk
   conda install traits
   pip install hermes-simulator

Examples
============

.. code-block:: python

    from hermes.examples import O3b_example

    O3b_example()
