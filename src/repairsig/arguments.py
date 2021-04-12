import configargparse
import logging
import sys, os
import pkg_resources

from repairsig import __version__

__author__ = "Hoinka, Jan"
__copyright__ = "Hoinka, Jan"
__license__ = "GPL-3.0-only"

_logger = logging.getLogger(__name__)

def check_positive_int(value, minval=0):
    """Validate integer input in arguments.

    Args:
      value: input value to check
      minval: value must be equal to or larger than minval

    Returns:
      True is sanity check passes, raises error otherwise
    """
    try:
        ivalue = int(value)
    except:
        raise configargparse.ArgumentTypeError(f"{value} is not an integer")

    if ivalue <= minval:
        raise configargparse.ArgumentTypeError(f"{value} is an invalid int value. Must be >= {minval}")

    return ivalue

def check_positive_float(value, minval=0.0):
    """Validate integer input in arguments.

    Args:
      value: input value to check
      minval: value must be equal to or larger than minval

    Returns:
      True is sanity check passes, raises error otherwise
    """
    try:
        fvalue = float(value)
    except:
        raise configargparse.ArgumentTypeError(f"{value} is not a float")

    if fvalue <= minval:
        raise configargparse.ArgumentTypeError(f"{value} is an invalid float value. Must be >= {minval}")

    return fvalue

def check_lower(value):
    """Converts the string to lower case

    Args:
        value: input value to convert

    Returns:
        Lower case represntation of value
    """

    return value.lower()

def check_upper(value):
    """Converts the string to upper case

    Args:
        value: input value to convert

    Returns:
        Upper case represntation of value
    """

    return value.upper()

def get_default_config_path():
    """Return the path on the file systems
    where the default config file is located as
    this might differ depending on where the
    package was installed.

    See: https://setuptools.readthedocs.io/en/latest/pkg_resources.html#basic-resource-access
    """

    resource_package = __name__
    resource_path = '/'.join(('data', 'default.conf'))
    file = pkg_resources.resource_filename(resource_package, resource_path)

    return(file)

def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = configargparse.ArgumentParser(default_config_files=[get_default_config_path()])

    #mandatory mutation count matrix
    parser.add('-M', '--matrix', required=True,  nargs='+', help='Mutation count matrix, specified as mutational profiles for each of the L genomic regions')

    #primary signatures. either P or N or both are required, but at least one
    p_group = parser.add_argument_group('Primary signature options (at least one option required)')
    p_group.add_argument('-P', '--primary', help='File containing the primary signature matrix of size NxK. If omitted, primary signatures will be infered according to -N.')
    p_group.add_argument('-N', '--numpri', type=check_positive_int,  help='Number of primary signatures to infer. If -P is specificed in addition to -N, RepairSig will infer N signatures while keeping the ones defined in P as static.')

    #secondary signatures
    q_group = parser.add_argument_group('Secondary signature options (at least one option required)')
    q_group.add_argument('-Q', '--secondary', help='File containing the primary signature matrix of size JxK')
    q_group.add_argument('-J', '--numsec', type=check_positive_int, help=f"Number of secondary signatures to infer.  If -Q is specified in addition to -J, RepairSig will infer J secondary signatures while keeping the ones in Q static. If J=0, and -Q is ommited, RepairSig defaults to the linear model.")

    #additional input options
    parser.add('-l', '--labels', default=False, action='store_true', help=f"Whether the input contains labels. If set, RepairSig will treat the first row/column of each input matrix as labels and use this information when generating the output files.")
    parser.add('-d', '--delimiter', default='\t', help="The delimiter used to separate the column in the input matrices. Default is tabulator. This delimiter will also be used for the output files.")

    #output options
    parser.add_argument('-O', '--out', required=False, default=os.getcwd(), help='Path to output folder. Folder will be created if it does not not exist. If omitted, the current folder will be used as the output directory. Existing files will be overwritten.')
    parser.add_argument('-v', '--verbosity', type=check_upper, default='INFO', choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], help='Set the verbosity level defining the detail of RepairSig messages to stdout.')

    #optimizer options
    q_group = parser.add_argument_group('Optimizer options (all or none are required)')
    q_group.add_argument('-e', '--epochs', nargs='+', type=check_positive_int, help='List of integers specifying the number of epochs in each optimization bracket. List must be of same size as -s and -o')
    q_group.add_argument('-s', '--stepsizes', nargs='+', type=check_positive_float, help=f"List of integers specifying the stepsize for the corresponding bracket defined with -e. List must be of same size as -e and -o")
    q_group.add_argument('-o', '--optimizers', nargs='+', type=check_lower, choices=["adadelta","adagrad","adam","adamax","nadam","rmaprop","sgd"], help=f"List of optimizers to use in each bracket defined with -e. List must be of same size as -e and -s")

    #config file
    parser.add('-c', '--config', required=False, is_config_file=True, help='Config file path')

    #hidden arguments, not visiable to the user but used internally by repairsig
    parser.add_argument('--optimizer_user_update_steps', type=check_positive_int, help=configargparse.SUPPRESS)
    parser.add_argument('--optimizer_log_update_steps',  type=check_positive_int, help=configargparse.SUPPRESS)




    args = parser.parse_args()

    #Perform additional sanity checks that are out of scope for the configargparse package
    #1) We need at least one primary signature option.
    if args.primary is None and args.numpri is None:
        parser.error(f"At least one primary signature option is required. Either use -P or -N, or both.")

    #2) We need at least one secondary signature option.
    if args.secondary is None and args.numsec is None:
        parser.error(f"At least one seconary signature option is required. Either use -Q or -J, or both.")

    #3) N must be greater or equal to 1
    if args.numpri is not None and args.numpri < 1:
        parser.error(f"The number of primary signatures (-N) must be >= 1.")

    #4) J must be greater or equal to 0
    if args.numsec is not None and args.numsec < 0:
        parser.error(f"The number of secondary signatures (-J) cannot be negative.")

    #5) Either all or none of the optimizer parameters must be specified. If they are, the lists must be of equal size
    opt_group = [args.epochs, args.stepsizes, args.optimizers]

    if sum(x is None for x in opt_group) not in [0,3]:
        parser.error(f"Either all or non of the opmizier parameters -e, -s, and -o must be specified.")

    if sum(x is not None for x in opt_group) == 3 and len(set(map(len, opt_group))) != 1:
        parser.error(f"The opmizier parameters must specify the same number of brackets. Currently the number of brackets for each option is as follows: epochs:{len(args.epochs)}, stepsizes:{len(args.stepsizes)}, optimizers:{len(args.optimizers)}")


    return args
