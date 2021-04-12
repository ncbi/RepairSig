"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = repairsig.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This skeleton file can be safely removed if not needed!

References:
    - https://setuptools.readthedocs.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import configargparse
import logging
import sys, os
import pkg_resources
import numpy as np

from repairsig import __version__
from repairsig import arguments
from repairsig import resources
from repairsig import optimizer
from repairsig import exporter

__author__ = "Hoinka, Jan"
__copyright__ = "Hoinka, Jan"
__license__ = "GPL-3.0-only"

_logger = logging.getLogger(__name__)



# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from repairsig.skeleton import check_positive_int`,
# when using this Python module as a library.


def setup_logging(args):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s: %(message)s"
    logging.basicConfig(
        level=logging.getLevelName(args.verbosity), stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.

def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formated message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    #parse the raw args
    args = arguments.parse_args(args)

    #setup the root logger
    setup_logging(args)

    #initialize tensors
    data = resources.Resources(args)


    #initialize optimizer
    opt = optimizer.Optimizer(data)
    #perform optimization
    opt.optimize()
    #store in resources
    opt.store()


    #export to file
    export = exporter.Exporter(data)
    export.write_tables()

    _logger.info(f"Completed. Exiting")

def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m repairsig.skeleton 42
    #
    run()
