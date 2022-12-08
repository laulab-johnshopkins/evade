# SCOOPS
SCOOPS analyzes binding site dynamics and allostery. It can find the volume of a pocket in an MD trajectory, suggest atom-atom distances as order parameters, and correlate an order parameter with dihedral motion.

## Installation
* Download this repository and `cd` into it.
* Run the following commands:

        pip install . -r requirements.txt
        pip uninstall ipykernel
        pip install ipykernel
        
### Notes
* The uninstall/reinstall of ipykernel is because of [this bug](https://github.com/jupyter-widgets/ipywidgets/issues/2257#issuecomment-1110226272) in a dependency.
* Another dependency fails when the ipywidgets version is too recent.  (This is documented [here](https://github.com/pyvista/pyvista/issues/3274).)  The SCOOPS requirements.txt automatically installs ipywidgets version 7.7.1, which is OK.  But if you install additional software after installing SCOOPS, then ipywidgets might get replaced.  This can be fixed with `pip install ipywidgets==7.7.1`.

## Usage
To see how to use SCOOPS, please read the Jupyter notebook in the `example` directory.

## Troubleshooting
* Issue: displayed proteins can't rotate/zoom.
  * Possible solution: uninstall/reinstall ipykernel
* Issue: displaying proteins takes too long.
  * Possible solutions: simplify the sel_regions, and/or increase grid_size so the surface uses fewer voxels.