Troubleshooting
===============

This document describes some common issues and how to fix them.

.. ipython:: python

    import volume_finder
    import MDAnalysis as mda

=======================================
Issue: no visualization window appears.
=======================================

Possible explanation: there is a known bug in a SOOPS dependency, as described
`here <https://github.com/pyvista/pyvista/issues/3274>`_.  To fix this, run the command
``pip install ipywidgets==7.7.1``.

========================================================================
Issue: while attempting to visualize a protein, the Jupyter kernel dies.
========================================================================

Possible explanation: When running on a remote server, run ``import pyvista as pv`` and
``pv.start_xvfb()``.