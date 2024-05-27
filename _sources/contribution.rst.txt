Contribute
==========

To contribute, please open a pull request to the :code:`dev`-branch on `GitHub <https://www.github.com/viljarjf/tiltlib/pulls>`_.

The following is an example of how to set up VS Code for development, adapt to your IDE of choice.

TL;DR:
------ 

:code:`pip install -e .`

Requirements
------------

- VS Code with the Python extension
- Python 3.8 or above

Setup
-----

1. Clone the repo and open it in VS Code.
2. Press f1, and run :code:`Python: Create Environment`. Select :code:`.venv`
3. Open a new terminal, which should automatically use the virtual environment. If not, run :code:`.venv\\Scripts\\activate` on Windows, or :code:`source .venv/bin/activate` on Unix
4. In the same terminal, run :code:`pip install -e .[test]` to install the current directory in an editable state, and the testing utility Pytest
5. To run tests, press f1 and run :code:`Python: Configure Tests`. Choose :code:`pytest`. Run tests in the testing menu
