# RepairSig

Reference implementation of RepairSig, a computational approach that accounts for the non-additivity of  DNA damage and repair processes by modeling the composition of primary mutagenic processes corresponding to DNA damage processes with normally functioning DNA repair mechanism and secondary mutagenic processes which correspond to the deficiency of the DNA repair mechanism. RepairSig assumes signatures of the primary processes are known while signatures of the secondary processes are to be inferred.

## Prerequisites

RepairSig requires Python 3.x and the following packages installed on your system:

* Tensorflow 2.x
* Numpy

## Usage
```bash
usage: python repairsig.py [-h] [--noweights]  [-o OUT] -i INPUT -J MMR

Arguments:
  -h, --help                show this help message and exit
  -J MMR, --MMR MMR         Number of expected MMR signatures.
  --noweights               No weights for genomic regions, i.e. set W and R to 1. Do not optimize W and R.
  -o OUT, --out OUT         path to the output file into which the trained tensors will be written
  -i INPUT, --input INPUT   path to the input file containing the given tensors
```

For example, to run RepairSig for `J=2` repair signatures and write the output file `repairsig.out`, use
```bash
python repairsig.py -J 2 -o repairsig.out -i input.txt
```

where `input.txt` is a tab delimited text file containing the flattened versions for input tensors `M` and `P` in the following format

```text
M     G,K,L      value1,value2, ... valueX
P     N,K        value1,value2, ... valueY
```

Here, `M` and `P` are the identifiers of the corresponding tensors and `G`, `K`, `L`, and `N` are to be replaced with the following dimensions
* `G` -> Number of genomes/patients/samples
* `K` -> Number of mutation categories (e.g. `K=96`)
* `L` -> Number of genomic regions
* `N` -> Number of mutational signatures

Note that `X=G*K*L` and `Y=N*K`.

## Optimization Strategy

RepairSig uses a default optimization strategy for its Adam optimizer as follows

|No. Epochs| Stepsize|
|----------|---------|
| 5000     |50       |
| 5000     |10 |
| 20000    | 1 |
| 20000    | 0.1|
| 5000     | 50|
| 5000     | 25|
| 5000     | 10 |
| 10000    | 0.1|
| 10000    | 0.01|
| 10000    | 0.005 |

It is possible to change this default behaviour by modifying the `config['optimizer_iterations']` and `config['optimizer_stepsize']` parameters in the python script respectively.
