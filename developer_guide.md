# Developer Guide
## Generating Documentation
Documentation is stored in the `doc` directory.  To recreate this directory, do the following:

* Run the command `sphinx-quickstart doc`.  The program will ask some questions; answer them.  Choose "yes" when asked about separating source and build directories.
* In `doc/source/conf.py`, change the following:
  * In the "Path setup" section (near the top), uncomment the `os` and `sys` imports.  Uncomment the `sys.path.insert` line, and change it to `sys.path.insert(0, os.path.abspath('../..'))`.
  * In the `extensions` list (which is initially empty), add `'sphinx.ext.autodoc'` and `'sphinx.ext.napoleon'`.
* Run the command `sphinx-apidoc -f -o doc/source .`
* Run the command `sphinx-build -b html doc/source doc/build/`.
* NOTES:
  * There will be an error about `modules.rst` not being in any toctree.  This can be fixed by changing `index.rst` to include the following (including whitespace):



               :caption: Contents:
          
               modules.rst
               
  * The output `index.html` might lack sufficient text.  If so, edit `index.rst`.
      * Under the `toctree` command, you can add filenames.  E.g. add `page1` if there is a `page1.rst` file.  Note that the rst file must have a title for autodoc to include it.

## Adding Functions To The Code
Whenever a new function is added, it must be imported in `__init__.py`.
