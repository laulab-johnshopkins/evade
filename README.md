[![CI Tests](https://github.com/laulab-johnshopkins/EVADE/actions/workflows/python-app.yml/badge.svg)](https://github.com/laulab-johnshopkins/EVADE/actions/workflows/python-app.yml)

# EVADE

EVADE analyzes binding site dynamics and allostery. It can find the volume of a pocket in an MD trajectory, suggest atom-atom distances as order parameters, and correlate an order parameter with dihedral motion.

## Installation
EVADE should work on any standard Mac or Linux workstation/laptop.  Installation should not take longer than a few minutes.
* Download this repository and `cd` into it.
* Run the following command:
    `pip install . -r requirements.txt`

        
### Notes
* A dependency fails when the ipywidgets version is too recent.  (This is documented [here](https://github.com/pyvista/pyvista/issues/3274).)  The EVADE requirements.txt automatically installs ipywidgets version 7.7.1, which is OK.  But if you install additional software after installing EVADE, then ipywidgets might get replaced.  This can be fixed with `pip install ipywidgets==7.7.1`.

## Usage
* To see how to use EVADE, please read the Jupyter notebooks in the `example` directory.
* These example notebooks should run in a matter of minutes (or sooner).
* For complete documentation, download EVADE and open `doc/build/index.html` in a web browser.

## Troubleshooting
* Issue: displaying proteins takes too long.
  * Possible solutions: simplify the sel\_regions, and/or increase grid\_size so the surface uses fewer voxels.
* Issue: the volumes code returns `KeyError`:
  * Solution: This probably means that the MDAnalysis universe does not have atom types.  Run this code to fix the issue for universe `u`:
  
      ```
      for atom in u_desres.atoms:
          atom.type = atom.name
      ```
