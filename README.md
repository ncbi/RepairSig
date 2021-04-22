# RepairSig

Reference implementation of RepairSig, a computational approach that
accounts for the non-additivity of  DNA damage and repair processes by
modeling the composition of primary mutagenic processes corresponding to
DNA damage processes with normally functioning DNA repair mechanism and
secondary mutagenic processes which correspond to the deficiency of the
DNA repair mechanism. RepairSig assumes signatures of the primary
processes are known while signatures of the secondary processes are to be
inferred.


## Prerequisites
RepairSig requires Python 3.x and the following packages installed on your system:

* Tensorflow 2.x
* Numpy
* Pandas
* Matplotlib
* Configargparse

## Installation
Clone the repository to a local folder, then use pip to install the package locally. RepairSig will be commited to PyPI soon.
```bash
git clone https://github.com/ncbi/RepairSig.git
cd RepairSig
pip install -e ./

repairsig -h
```

## Usage
```bash
repairsig -h
usage: repairsig [-h] -M MATRIX [MATRIX ...] [-P PRIMARY] [-N NUMPRI]
                 [-Q SECONDARY] [-J NUMSEC] [-l] [-d DELIMITER] [-O OUT]
                 [-v {CRITICAL,ERROR,WARNING,INFO,DEBUG}]
                 [-e EPOCHS [EPOCHS ...]] [-s STEPSIZES [STEPSIZES ...]]
                 [-o {adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd} [{adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd} ...]]
                 [-c CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -M MATRIX [MATRIX ...], --matrix MATRIX [MATRIX ...]
                        Mutation count matrix, specified as mutational
                        profiles for each of the L genomic regions
  -l, --labels          Whether the input contains labels. If set, RepairSig
                        will treat the first row/column of each input matrix
                        as labels and use this information when generating the
                        output files.
  -d DELIMITER, --delimiter DELIMITER
                        The delimiter used to separate the column in the input
                        matrices. Default is tabulator. This delimiter will
                        also be used for the output files.
  -O OUT, --out OUT     Path to output folder. Folder will be created if it
                        does not not exist. If omitted, the current folder
                        will be used as the output directory. Existing files
                        will be overwritten.
  -v {CRITICAL,ERROR,WARNING,INFO,DEBUG}, --verbosity {CRITICAL,ERROR,WARNING,INFO,DEBUG}
                        Set the verbosity level defining the detail of
                        RepairSig messages to stdout.
  -c CONFIG, --config CONFIG
                        Config file path

Primary signature options (at least one option required):
  -P PRIMARY, --primary PRIMARY
                        File containing the primary signature matrix of size
                        NxK. If omitted, primary signatures will be infered
                        according to -N.
  -N NUMPRI, --numpri NUMPRI
                        Number of primary signatures to infer. If -P is
                        specificed in addition to -N, RepairSig will infer N
                        signatures while keeping the ones defined in P as
                        static.

Secondary signature options (at least one option required):
  -Q SECONDARY, --secondary SECONDARY
                        File containing the primary signature matrix of size
                        JxK
  -J NUMSEC, --numsec NUMSEC
                        Number of secondary signatures to infer. If -Q is
                        specified in addition to -J, RepairSig will infer J
                        secondary signatures while keeping the ones in Q
                        static. If J=0, and -Q is ommited, RepairSig defaults
                        to the linear model.

Optimizer options (all or none are required):
  -e EPOCHS [EPOCHS ...], --epochs EPOCHS [EPOCHS ...]
                        List of integers specifying the number of epochs in
                        each optimization bracket. List must be of same size
                        as -s and -o
  -s STEPSIZES [STEPSIZES ...], --stepsizes STEPSIZES [STEPSIZES ...]
                        List of integers specifying the stepsize for the
                        corresponding bracket defined with -e. List must be of
                        same size as -e and -o
  -o {adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd} [{adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd} ...], --optimizers {adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd} [{adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd} ...]
                        List of optimizers to use in each bracket defined with
                        -e. List must be of same size as -e and -s

Args that start with '--' (eg. --matrix) can also be set in a config file
(./src/repairsig/data/default.conf or specified via -c). Config
file syntax allows: key=value, flag=true, stuff=[a,b,c] (for details, see
syntax at https://goo.gl/R74nmi). If an arg is specified in more than one
place, then commandline values override config file values which override
defaults. 
```


## Example

To apply RepairSig on BRCA whole-genome sequencing data (Nik-Zainal et al., Nature 2016) with local regional activity using

* strand-specific direction of gene transcription, run
```
repairsig -c test_data/BRCA_T.conf -J 2 -O output_T
```
* discretized replication timing data, run
```
repairsig -c test_data/BRCA_RT.conf -J 2 -O output_RT
```

Most input and parameters of the model (e.g. mutation counts, primary signatures) are defined in the provided config files via -c.
The number of secodary signatures to be inferred, here, is 2.

## Citation
[Damian Wojtowicz, Jan Hoinka, Bayarbaatar Amgalan, Yoo-Ah Kim, Teresa M. Przytycka, ''RepairSig: Deconvolution of DNA damage and repair contributions to the mutational landscape of cancer'', bioRxiv, 2020](https://www.biorxiv.org/content/10.1101/2020.11.21.392878v1)

## Note
This project has been set up using PyScaffold 4.0. For details and usage
information on PyScaffold see https://pyscaffold.org/.
