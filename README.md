# RepairSig

Reference implementation of RepairSig, a computational approach that accounts for the non-additivity of  DNA damage and repair processes by modeling the composition of primary mutagenic processes corresponding to DNA damage processes with normally functioning DNA repair mechanism and secondary mutagenic processes which correspond to the deficiency of the DNA repair mechanism. RepairSig assumes signatures of the primary processes are known while signatures of the secondary processes are to be inferred.


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
pip install ./
```

## Note

This project has been set up using PyScaffold 4.0. For details and usage
information on PyScaffold see https://pyscaffold.org/.
