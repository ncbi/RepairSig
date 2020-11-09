#!/bin/env python

import sys,os,glob,re
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import numpy as np
from numpy import genfromtxt

#define arguments required for this script
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-J', '--MMR', default=2, required=True, help='Number of expected MMR signatures.')
parser.add_argument('--noweights', action='store_true', default=False, required=False, help='No weights for genomic regions, i.e. set W and R to 1. Do not optimalize W and R.')
parser.add_argument('-t', '--tensorboard', required=False, help='path to tensorboard folder. tensorboard logging will be disabled if not specified')
parser.add_argument('-s', '--logsuffix', required=False, help='optional suffix to attach to the log counter in tensorboard')
parser.add_argument('-o', '--out', required=False, help='path to the output file into which the trained tensors will be written')
parser.add_argument('-i', '--input', required=True, help='path to the input file containing the given tensors')

args = parser.parse_args()

# creates a configuration
def getConfig():
    config = {}

    # get the tensorboard logging folder if specified
    if args.tensorboard != None:

        logsuffix = "" if args.logsuffix == None else "_"+args.logsuffix

        maxnum = 0
        for the_file in os.listdir(args.tensorboard+"/"):
            if os.path.isdir(args.tensorboard+"/"+the_file):
                    try:
                        counter_prefix = re.search(r'\d+', the_file).group()
                        maxnum = max(maxnum, int(counter_prefix))
                    except:
                        pass

        config['tensorboard'] = os.path.abspath(args.tensorboard+"/"+str(maxnum+1)+logsuffix)

    else:
        config['tensorboard'] = None

    if args.out != None:
        config['output'] = os.path.abspath(args.out)
    else:
        config['output'] = None

    config['input'] =  os.path.abspath(args.input)

    # Add data dimensions
    config['G'] = None  # Number of patient, samples, genomes
    config['L'] = None    # Number of genomic regions, e.g. early/late replication region, inter/intragenic
    config['K'] = None    # Number of mutations categories, e.g. A[C>T]G, C[T>A]G
    config['N'] = None    # Number of PRIMARY mutational signatures
    config['J'] = int(args.MMR)    # Number of DNA REPAIR mutational signatures. Given as input.

    # Are regionsweighted?
    config['noweights'] = args.noweights

    # Add ADAM specific properties
    config['optimizer_iterations'] = [5000,5000,20000,20000,5000,5000,5000,5000,10000,10000,10000]
    config['optimizer_stepsize'] =   [  50,  10,    1,  0.1,  50,  25,  10,  10,  0.1, 0.01,0.005]
    config['optimizer_user_update_steps'] = 500     #Number of updates to print during optimization

    return config

# plots the loaded model into tensorboard, has no effect if no tensorboard
# folder was specified
def graphModel( ):
    if config['tensorboard'] != None:

        #because we are executing egerly, no graph is being
        #created. Hence, we need to manually initiate a trace
        #before we can visualize the computation
        tf.summary.trace_on(graph=True)
        foo_g = tf.function(compute_loss)
        foo_g()

        #now we can write the graph to tensorboard
        writer = config['writer']
        with writer.as_default():
            tf.summary.trace_export(
                name='tf2_graph',
                step=0
            )

# Writes all trainable tensors to file.
# If config['ouput'] == None, this function has no effect.
# One line per tensor, tab separated. [ID, Shape, Serialized Data]
# Serialized Data and Shape are comma separated internally, eg
# Tensor1    2,2    1,2,3,4
# Tensor2    2,3    1,2,3,4,5,6
def exportData(variables):
    if config['output'] != None:
        with open(config['output'], 'w') as file:
            for name in variables:
                tensor_py = eval(name).numpy()

                #write ID:
                file.write(name)
                file.write("\t")

                #write shape
                file.write(','.join(map(lambda x: str(x),tensor_py.shape)) )
                file.write("\t")

                #write serialized tensor
                file.write( ','.join(map(lambda x: str(x), tensor_py.reshape([-1]))) )
                file.write("\n")


# See https://stackoverflow.com/questions/59309114/tensor-n-mode-product-in-tensorflow
def n_mode_product(x, u, n, name=None):
    n = int(n)
    # We need one letter per dimension
    # (maybe you could find a workaround for this limitation)
    if n > 26:
        raise ValueError('n is too large.')
    ind = ''.join(chr(ord('a') + i) for i in range(n))
    exp = f"{ind}K...,JK->{ind}J..."
    return tf.einsum(exp, x, u, name=name)



def readInputData(verbose=False):
    #get data from file
    data = {}
    with open(config['input']) as file:
        for line in file:
            line = line.split()
            mat = np.array(list(map(lambda x: float(x), line[2].split(',')))) #matrix or tensor
            shape = list(map(lambda x: int(x), line[1].split(','))) #matrix/tensor shape
            mat = mat.reshape(shape)
            name = line[0]
            #save the tensor in data dict
            data[name] = mat

    if verbose:
        print(f"Read {len(data)} matrices: {','.join(data.keys())}")

    return data


# Function generating input tensor P.
def getTensor(name, verbose=False):

    T = tf.Variable(
                    initial_value=tf.convert_to_tensor(value=data[name]),
                    trainable=False,
                    name=name+"_input"
                    )

    if verbose:
        print(f"Got {name} of shape {T.shape}, min {tf.reduce_min(T)}, max {tf.reduce_max(T)}")

    return T


def getRandomTensor(shape, min, max, trainable=True, clip_min=np.NINF, clip_max=np.infty, sum_one=False, sum_one_axis=None, name=None, verbose=False):
    """Function generating a random tensor of shape containing values.

    Parameters
    ----------
    shape : array or tuple
        The shape of the tensor to generate
    min : float
        smallest element in tensor (initial value)
    max : float
        largest element in tensor (initial value)
    trainable : boolean
        Whether this tensor should be optimized
    clip_min: float
        If optimized, any values below clip_min will be clipped
    clip_max: float
        If optimized, any values above clip_max will be clipped
    sum_one : boolean
        Whether to constain this tensor to have all elements sum to one
    sum_one_axis : array or integer
        If sum_one is true, it reduces this tensor along the dimensions given in axis
    name : string
        Description of the tensor
    verbose: bool
        If true, print message regarding shape of the tensor

    Returns
    -------
    tf.Tensor
        Tensor with the properties as described by the parameters

    """

    tensor = np.random.uniform(size=shape) * (max - min) + min

    #constraints
    if sum_one: #probabilities along sum_one_axis
        constraint = lambda x: tf.clip_by_value(x,clip_min,clip_max) / tf.reduce_sum(tf.clip_by_value(x,clip_min,clip_max), axis=sum_one_axis, keepdims=True)
    else:
        constraint = lambda x: tf.clip_by_value(x, clip_min,clip_max)

    tensor =  tf.Variable(
                        initial_value=tf.convert_to_tensor(value=tensor),
                        trainable=trainable,
                        constraint=constraint,
                        name=name)

    if verbose:
        print(f"Generated initial {name} of shape {tensor.shape}, min {tf.reduce_min(tensor)}, max {tf.reduce_max(tensor)}")

    return tensor


def optimizeModel(loss, iters, stepsize):

    #print directives
    update_steps = tf.constant(config['optimizer_user_update_steps'])
    epoch = tf.Variable(0)
    num_it = tf.constant(iters)
    cont = tf.Variable(True)
    #current best loss
    best_loss = tf.Variable(tf.cast(loss(), tf.float32))
    #best parameters so far
    W_best = tf.Variable(tf.identity(W), shape=W.shape)
    A_best = tf.Variable(tf.identity(A), shape=A.shape)
    Q_best = tf.Variable(tf.identity(Q), shape=Q.shape)
    R_best = tf.Variable(tf.identity(R), shape=R.shape)
    D_best = tf.Variable(tf.identity(D), shape=D.shape)

    # Define a training operation for tensforflow, this can be exchanged with other optimizers if desired
    train_op = tf.optimizers.Adam(stepsize)
    #train_op = tf.optimizers.Nadam(stepsize)
    #train_op = tf.optimizers.Adadelta(stepsize)

    @tf.function
    def doOpt():
        while epoch < num_it and cont:

            # evaluate graph. this will compute gradients, and update trainable
            # variables under the hood
            train = train_op.minimize(loss, trainable_variables)
            loss_val = tf.cast(loss(), tf.float32)

            # store the best weights so far, these are the ones we will return
            tf.cond(
                tf.less(loss_val,best_loss),
                lambda: [W_best.assign(W), A_best.assign(A), Q_best.assign(Q), R_best.assign(R), D_best.assign(D), best_loss.assign(loss_val)],
                lambda: [W_best, A_best, Q_best, R_best, D_best, loss_val] ##dummy for the graph
            )

            epoch.assign(epoch + 1)

            # print update?
            tf.cond(tf.equal(epoch % update_steps, 0),
                    lambda: [   #True
                                tf.print("Epoch", epoch, "loss", loss_val, output_stream=sys.stdout) #print update message
                            ],
                    lambda: [   #False
                                tf.no_op() #just a placeholder
                            ])


    # Perform optimization
    tf.print("Epoch", 0, "loss", tf.cast(loss(), tf.float32), output_stream=sys.stdout)
    doOpt()

    # Save best results at the end of the current iteration
    W.assign(W_best)
    A.assign(A_best)
    Q.assign(Q_best)
    R.assign(R_best)
    D.assign(D_best)


# Define a loss. We use the Frobenuis loss.
# This function will be wrapped into a GradientTape during eager
# execution
def compute_loss():
    # Mhat = term1 * term2 + term3

    term1 = n_mode_product(tf.multiply(P,W), A, 0)
    term2 = 1 + n_mode_product(tf.multiply(Q,R), D, 0)

    Mhat = term1 * term2

    loss_value = tf.norm(
                tensor=M-Mhat,
                ord='euclidean',
                name='frobenius_norm'
                )

    return loss_value

###############################################################################
#MAIN
verbose=True
minval = 1e-10
maxval = 1000

config = getConfig()


# Create logging folder if required
if config['tensorboard'] != None:
    os.mkdir(config['tensorboard'])
    config['writer'] = tf.summary.create_file_writer(config['tensorboard'])
    print("Enabled tensorboard logging to folder %s" % config['tensorboard'])
else:
    config['writer'] = None
    print("No tensorboard directory specified, disabling logging.")

#read input matrices
data = readInputData(verbose=verbose)

# Initialise tensors
# Tensor ranks are:
# M -> G x K x L (input) - mutation counts of category k in region l of genome g
# P -> N x K x 1 (input) - PRIMARY mutational signature matrix (N signatures, K mutation categories)
# P -> N x K x 1 (input) - PRIMARY mutational signature matrix (N signatures, K mutation categories)
# W -> N x 1 x L (trainable or constant) - regional activity matrix for PRIMARY signatures (regional weights)
# A -> G x N (trainable) - genome activity matrix for PRIMARY signatures
# Q -> J x K x 1 (trainable) - MMR mutational signature matrix
# R -> J x 1 x L (trainable or constant) - regional activity matrix for MMR signatures (regional weights)
# D -> G x J (trainable) - genome activity matrix for MMR signatures

# Adding background signature
# PB -> 1 x K x 1 (input)
# P <- tf.concat(P,PB)
# N <- N + 1

# input mutation count matrix
M = getTensor('M', verbose=verbose)

# PRIMARY mutational signature matrix + BACKGROUND signature
Pprimary = getTensor('P', verbose=verbose)
Pbackground = getTensor('PB', verbose=verbose)
P = tf.concat([Pprimary,Pbackground],0)
P = tf.reshape(P, P.shape+[1])

# set config dimentions
config['G'], config['K'], config['L'] = M.shape
config['N'] = P.shape[0] #number of primary signatures + 1 BG signature

# regional activity matrix for PRIMARY signatures (regional weights)
W = getRandomTensor(shape=[config['N'],1,config['L']],
                    min=minval if not config['noweights'] else 1.0,
                    max=1.0,
                    trainable=not config['noweights'],
                    clip_min=minval,
                    sum_one=True,
                    sum_one_axis=2, #reduce sum over L (3rd dim)
                    name="W",
                    verbose=verbose)
# genome activity matrix for PRIMARY signatures
A = getRandomTensor(shape=[config['G'],config['N']],
                    min=0,
                    max=maxval,
                    trainable=True,
                    clip_min=0,
                    name="A",
                    verbose=verbose)
# MMR mutational signature matrix
Q = getRandomTensor(shape=[config['J'],config['K'],1],
                    min=minval,
                    max=1.0,
                    trainable=True,
                    clip_min=minval,
                    sum_one=True,
                    sum_one_axis=1, #reduce sum over K (2nd dim)
                    name="Q",
                    verbose=verbose)
# regional activity matrix for MMR signatures (regional weights)
R = getRandomTensor(shape=[config['J'],1,config['L']],
                    min=minval if not config['noweights'] else 1.0,
                    max=1.0,
                    trainable=not config['noweights'],
                    clip_min=minval,
                    sum_one=True,
                    sum_one_axis=2, #reduce sum over L (3rd dim)
                    name="R",
                    verbose=verbose)
# genome activity matrix for MMR signatures
D = getRandomTensor(shape=[config['G'],config['J']],
                    min=0,
                    max=maxval,
                    trainable=True,
                    clip_min=0,
                    name="D",
                    verbose=verbose)


#Define which tensors should be optimized
trainable_variables = [W,A,Q,R,D] if not config['noweights'] else [A,Q,D]

print(f"> Config: {config}")

print("> Model definition complete, commencing optimization")
t0= timer()
# Perform optimization
for iters,stepsize in zip(config['optimizer_iterations'],config['optimizer_stepsize']):
    tf.print("### New optimization: iterations", iters, "step size", stepsize, output_stream=sys.stdout)
    optimizeModel(compute_loss, iters, stepsize)
    tf.print("Best loss", tf.cast(compute_loss(), tf.float32), output_stream=sys.stdout)
t1 = timer()

print(f"> Optimization complete. Time elapsed: {t1 - t0} seconds\n") # CPU seconds elapsed (floating point)

#post processing
graphModel( )
exportData(['W','A','Q','R','D'])
