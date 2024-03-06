# Developer Guide

## Table of Contents
[Generating Documentation](#gen_docs)

[Generating Requirements.txt](#gen_reqs)

[Imports](#imports)

[Testing](#testing)
## <a name="gen_docs"></a>Generating Documentation
Documentation is stored in the `doc` directory.  There are two parts:

1. API docs listing each function.  These are automatically generated from docstrings in the code.

2. Documents written by the developer (me).

The initial directory structure of the `doc` directory is created automatically; then the developer adds to it.

### Recreating the Directory Structure

To recreate this directory, do the following:

* Run the command `sphinx-quickstart doc`.  The program will ask some questions; answer them.  Choose "yes" when asked about separating source and build directories.
* In `doc/source/conf.py`, change the following:
  * In the "Path setup" section (near the top), uncomment the `os` and `sys` imports.  Uncomment the `sys.path.insert` line, and change it to `sys.path.insert(0, os.path.abspath('../..'))`.
  * In the `extensions` list (which is initially empty), add `'sphinx.ext.autodoc'` and `'sphinx.ext.napoleon'` and `'IPython.sphinxext.ipython_console_highlighting'` and `'IPython.sphinxext.ipython_directive'`.
* Run the command `sphinx-apidoc -f -o doc/source .`

### Adding Pages
* Each additional page must be included in a toctree.  E.g. if a page is called `new.rst`, then a toctree would look like this:

        .. toctree::
           :maxdepth: 2
           :caption: Contents:

           new.rst


### Creating the HTML Files

* Run the command `sphinx-build -b html doc/source doc/build/`.

### NOTES:
  * There will be an error about `modules.rst` not being in any toctree.  This can be fixed by changing `index.rst` to include the following (including whitespace):



               :caption: Contents:
          
               modules.rst
               
  * The output `index.html` might lack sufficient text.  If so, edit `index.rst`.
      * Under the `toctree` command, you can add filenames.  E.g. add `page1` if there is a `page1.rst` file.  Note that the rst file must have a title for autodoc to include it.q
  * The relative imports in my modules (the `.` in `from .module1 import *`) are needed to get Sphinx to not complain.

## <a name="gen_reqs"></a>Generating Requirements.txt
* `poetry export --without-hashes -f requirements.txt --output requirements.txt`
  * I would like to include hashes in requirements.txt.  This would increase security during software installation.  However, as described [here](https://github.com/python-poetry/poetry/issues/7122), poetry doesn't create hashes for some libraries.  This was observed to be an issue for a few DAVEE dependencies.

## <a name="imports"></a>Imports
Code is listed in `__init__.py`.  This tells Python where to look for the code.  If `__init__.py` were blank, then users would need to explicitly import each submodule (`from a_module import a_submodule`).  Including code in `__init__.py` allows users to import all code by importing the base module.

Prototype versions of the software had everything in a single file.  Each function was imported in `__init__.py` (e.g. `from .volume_finder import show_pocket`).  Now each submodule is listed in `__init__.py`.

## <a name="testing"></a>Testing
To run the tests:

* Install Astropy and pytest.
* Run the command `pytest`.  The result will probably include warnings, but all tests should pass.

To check code coverage:

* Install the `coverage` Python package.
* Run the command `coverage run -m pytest` then `coverage report`.  Replacing the second command with `coverage html` will give more detailed info.