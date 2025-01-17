# PyNBoids
<img src="/preview.gif" width="250" height="200"/>

### A Python Boids Simulation
This is a [Boids simulation](https://en.wikipedia.org/wiki/Boids 
"Wikipedia"),
written in Python3 using Pygame2, vectorized with PyTorch.

**To use:** Run via python. (Example: `python3 vectorboids.py`)

`Esc` key to quit.

### Simulation Settings
There are variety of customizable settings at the top of the `vectorboids.py` file. 

These are largely split into UI settings (screen size, frames/second) and simulation parameters (number of agents, neighborhood radius).

### Credits and Licensing
The framework code (game engine and UI) was written by Nikolaus Stromberg and licensed under the GNU General Public License (see `LICENSE_FRAMEWORK.md`)
* Copyright (c) 2021  Nikolaus Stromberg - nikorasu85@gmail.com

The ensemble code (vectorization of flocking) was written by Laura Sisson and licensed under the MIT License  (see `LICENSE_ENSEMBLE.md`)
* Copyright (c) 2023  Laura Sisson

