import os,sys
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import tensorflow as tf

#get logger instance
import logging
_logger = logging.getLogger(__name__)

class Optimizer(object):

    def __init__(self, data):

        _logger.debug(f"Creating instance of Optimizer")

        self.data = data
        self.args = data.args

        #Get a tf.Variable representation of all data required for the optimization
        self.M = tf.Variable(initial_value=tf.convert_to_tensor(value=data.get_m()),
                            trainable=False,
                            name='M')

        self.P_fixed, self.P_inferred = self.make_p()

        self.W = tf.Variable(initial_value=tf.convert_to_tensor(value=data.get_w()[:, np.newaxis, :]),
                            trainable=True,
                            constraint=lambda x: tf.clip_by_value(x,1e-10,np.infty) / tf.reduce_sum(tf.clip_by_value(x,1e-10,np.infty), axis=2, keepdims=True),
                            name='W')

        self.A = tf.Variable(initial_value=tf.convert_to_tensor(value=data.get_a()),
                            trainable=True,
                            constraint=lambda x: tf.clip_by_value(x,0,np.infty),
                            name='A')

        self.Q_fixed, self.Q_inferred = self.make_q()

        self.R = tf.Variable(initial_value=tf.convert_to_tensor(value=data.get_r()[:, np.newaxis, :]),
                            trainable=True,
                            constraint=lambda x: tf.clip_by_value(x,1e-10,np.infty) / tf.reduce_sum(tf.clip_by_value(x,1e-10,np.infty), axis=2, keepdims=True),
                            name='R')

        self.D = tf.Variable(initial_value=tf.convert_to_tensor(value=data.get_d()),
                            trainable=True,
                            constraint=lambda x: tf.clip_by_value(x,0,np.infty),
                            name='D')

        _logger.debug(f"Completed tf.Variable initializtion")

        #Define which tensors should be optimized
        self.trainable_variables = list(filter( lambda x: x is not None, [self.P_inferred, self.W, self.A, self.Q_inferred, self.R, self.D]))

        _logger.debug(f"Trainable variables are {[x.name for x in self.trainable_variables]}")

        #store performance values of interest here
        self.logs =  {
                    'losses' : tf.Variable([], dtype=tf.float32),
                    'best_losses' : tf.Variable([], dtype=tf.float32),
                    'epochs' : tf.Variable([], dtype=tf.int32),
                    'steps' : tf.Variable([], dtype=tf.float32)
                }

        _logger.info(f"Successfully initialized optimizer.")

    def make_p(self):
        """ Creates a Variable of P with corresponding contraints

            Array corrsponding to the fixed and inferred portion of P
        """

        _logger.debug(f"make_p")

        # get the fixed and inferred portion and combine
        p = self.data.get_p()

        p_fixed = p[0] # this is either None or not
        if p_fixed is not None: #Convert to Variable
            p_fixed = tf.Variable(
                                    initial_value=tf.convert_to_tensor(value=p[0][:, :, np.newaxis]), #reshape to NxKx1
                                    trainable=False,
                                    name='P_fixed'
                                )


        p_inferred = p[1]
        if p_inferred is not None:
            #constraint: probabilities must sum to one along the mutational signature axis (1)
            clip_min = 1e-10
            clip_max = np.infty
            sum_one_axis = 1
            constraint = lambda x: tf.clip_by_value(x,clip_min,clip_max) / tf.reduce_sum(tf.clip_by_value(x,clip_min,clip_max), axis=sum_one_axis, keepdims=True)

            p_inferred = tf.Variable(
                                initial_value=tf.convert_to_tensor(value=p[1][:, :, np.newaxis]),
                                trainable=True,
                                constraint=constraint,
                                name='P_inferred')

        return [p_fixed, p_inferred]

    def make_q(self):
        """ Creates a Variable of Q with corresponding contraints

            Array corrsponding to the fixed and inferred portion of Q
        """

        _logger.debug(f"make_q")

        # get the fixed and inferred portion and combine
        q = self.data.get_q()

        q_fixed = q[0] # this is either None or not
        if q_fixed is not None: #Convert to Variable
            q_fixed = tf.Variable(
                                    initial_value=tf.convert_to_tensor(value=q[0][:, :, np.newaxis]), #reshape to JxKx1
                                    trainable=False,
                                    name='Q_fixed'
                                )


        #constraint: probabilities must sum to one along the mutational signature axis (1)
        q_inferred = q[1]
        if q_inferred is not None:
            clip_min = 1e-10
            clip_max = np.infty
            sum_one_axis = 1
            constraint = lambda x: tf.clip_by_value(x,clip_min,clip_max) / tf.reduce_sum(tf.clip_by_value(x,clip_min,clip_max), axis=sum_one_axis, keepdims=True)

            q_inferred = tf.Variable(
                                initial_value=tf.convert_to_tensor(value=q[1][:, :, np.newaxis]),
                                trainable=True,
                                constraint=constraint,
                                name='Q_inferred')

        return [q_fixed, q_inferred]

    def n_mode_product(self, x, u, n, name=None):
        """ Tensorflow implementation of the n mode product.
         See https://stackoverflow.com/questions/59309114/tensor-n-mode-product-in-tensorflow
        """
        n = int(n)

        # We need one letter per dimension
        # (maybe you could find a workaround for this limitation)
        if n > 26:
            raise ValueError('n is too large.')
        ind = ''.join(chr(ord('a') + i) for i in range(n))
        exp = f"{ind}K...,JK->{ind}J..."

        return tf.einsum(exp, x, u, name=name)


    def compute_loss(self):
        """ Returns an objective funtion corresponding to the chosen settings in args

        """

        # concatenate P and Q depending on whether they are defined or not
        P = tf.concat(list(filter(lambda x: x is not None, [self.P_fixed, self.P_inferred])), axis=0)
        Q = tf.concat(list(filter(lambda x: x is not None, [self.Q_fixed, self.Q_inferred])), axis=0)

        # LINEAR MODEL CASE
        if self.args.secondary is None and self.args.numsec == 0:


            PQ = tf.concat([P,Q],0)
            WR = tf.concat([self.W,self.R],0)
            AD = tf.concat([self.A,self.D],1)

            Mhat = n_mode_product(tf.multiply(PQ,WR), AD, 0)

            loss_value = tf.norm(
                       tensor=self.M-Mhat,
                       ord='euclidean',
                       name='frobenius_norm'
                       )

            return loss_value

        # NONLINER MODEL
        term1 = self.n_mode_product(tf.multiply(P,self.W), self.A, 0)
        term2 = 1 + self.n_mode_product(tf.multiply(Q,self.R), self.D, 0)
        Mhat = term1 * term2

        loss_value = tf.norm(
                    tensor=self.M-Mhat,
                    ord='euclidean',
                    name='frobenius_norm'
                    )

        return loss_value

    def optimize(self):
        """
        Contains logic to perform the full optimization
        """

        _logger.info("Model definition complete, commencing optimization")
        t0= timer()
        # Perform optimization
        for epoch,stepsize,optimizer in zip(self.args.epochs,self.args.stepsizes, self.args.optimizers):
            _logger.info(f"### New optimization: iterations {epoch} step size {stepsize} optimizer {optimizer}")
            self.optimizeModel(self.compute_loss, epoch, stepsize, optimizer)
            _logger.info(f"Best loss {self.compute_loss()}")
        t1 = timer()

        _logger.info(f"Optimization complete. Time elapsed: {t1 - t0} seconds") # Wall seconds elapsed (floating point)



    def optimizeModel(self, loss, iters, stepsize, type):
        """
        Performs one optimization bracket according to the provided options

        Args:
            loss: identifier of which loss function to use
            iters: number of iterations (epochs) to perform
            stepsize: initial stepsize for the optimizer
            type: the optimizer type to use
        """

        _logger.debug(f"Preparing optimization")

        # Convert everything into tensorfow equivalents so we can run it on the GPU

        # print directives
        update_steps = tf.constant(self.args.optimizer_user_update_steps)
        log_update_steps = tf.constant(self.args.optimizer_log_update_steps)

        # optimization details
        epoch = tf.Variable(0)
        num_it = tf.constant(iters)
        cont = tf.Variable(True)

        # current best loss
        best_loss = tf.Variable(tf.cast(loss(), tf.float32))

        # best parameters so far. Since these change depending on the user input,
        # use an array to store them. They should always correspond to the
        # content and order of self.trainable_variables
        parameters_best = [tf.Variable(tf.identity(x), shape=x.shape) for x in self.trainable_variables]

        # Define a training operation for tensforflow, this can be exchanged with other optimizers if desired
        # adadelta,adagrad,adam,adamax,nadam,rmaprop,sgd
        if type == "adadelta":
            train_op = tf.optimizers.Adadelta(stepsize)
        elif type ==  "adagrad":
            train_op = tf.optimizers.Adagrad(stepsize)
        elif type ==  "adam":
            train_op = tf.optimizers.Adam(stepsize)
        elif type ==  "adamax":
            train_op = tf.optimizers.Adamax(stepsize)
        elif type ==  "nadam":
            train_op = tf.optimizers.Nadam(stepsize)
        elif type ==  "rmaprop":
            train_op = tf.optimizers.RMSprop(stepsize)
        elif type ==  "sgd":
            train_op = tf.optimizers.SDG(stepsize)
        elif type ==  _:
            train_op = tf.optimizers.Adam(stepsize)

        _logger.debug(f"Optimizer is {train_op}")

        @tf.function
        def doOpt():

            #temporary variables to store log
            loss_log_temp  = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            best_loss_log_temp = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            epoch_log_temp = tf.TensorArray(tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            steps_log_temp = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

            while epoch < num_it and cont:

                # evaluate graph. this will compute gradients, and update trainable
                # variables under the hood
                train = train_op.minimize(loss, self.trainable_variables)
                loss_val = tf.cast(loss(), tf.float32)

                # store the best weights so far, these are the ones we will return
                if loss_val < best_loss:
                    best_loss.assign(loss_val)
                    for i,x in enumerate(self.trainable_variables):
                        parameters_best[i].assign(x)

                # update logs?
                if epoch % log_update_steps == 0:
                    loss_log_temp = loss_log_temp.write(loss_log_temp.size(), loss_val)
                    best_loss_log_temp = best_loss_log_temp.write(best_loss_log_temp.size(), best_loss)
                    epoch_log_temp = epoch_log_temp.write(epoch_log_temp.size(), epoch)
                    steps_log_temp = steps_log_temp.write(steps_log_temp.size(), stepsize)

                # print update?
                tf.cond(tf.equal(epoch % update_steps, 0),
                        lambda: [   #True
                                    tf.print("Epoch", epoch, "loss", loss_val, "best loss", best_loss, output_stream=sys.stdout)
                                ],
                        lambda: [   #False
                                    tf.no_op() #just a placeholder
                                ])

                epoch.assign(epoch + 1)

                #hack or not, I could not find another way. We need to add the performance stats to the log, so we do it at the last iteration.
                #Since this code runs on the GPU, it will not be available after the for loop
                if( epoch+1 == num_it ):
                    self.logs['losses'].assign( tf.concat([self.logs['losses'], loss_log_temp.stack()], axis=0) )
                    self.logs['best_losses'].assign( tf.concat([self.logs['best_losses'], best_loss_log_temp.stack()], axis=0) )
                    self.logs['epochs'].assign( tf.concat([self.logs['epochs'], epoch_log_temp.stack()], axis=0) )
                    self.logs['steps'].assign( tf.concat([self.logs['steps'], steps_log_temp.stack()], axis=0) )


        # Perform optimization
        tf.print("Epoch", 0, "loss", tf.cast(loss(), tf.float32), output_stream=sys.stdout)
        doOpt()

        # Save best results at the end of the current iteration
        for x,y in zip(parameters_best, self.trainable_variables):
            y.assign(x)

        _logger.debug(f"Optimization bracket completed")


    def store(self):
        """
        Stores the optimized tensors back in the resources class. Takes care of
        removing the extra dimensions

        """
        _logger.info(f" Storing optimized parameters")

        self.data.set_p([
                        None if self.P_fixed is None else np.reshape(self.P_fixed.numpy(), self.P_fixed.shape[:-1]),
                        None if self.P_inferred is None else np.reshape(self.P_inferred.numpy(), self.P_inferred.shape[:-1])
                        ])

        self.data.set_w(np.reshape(self.W.numpy(), [self.W.shape[0], self.W.shape[-1]]))

        self.data.set_a(self.A.numpy())

        self.data.set_q([
                        None if self.Q_fixed is None else np.reshape(self.Q_fixed.numpy(), self.Q_fixed.shape[:-1]),
                        None if self.Q_inferred is None else np.reshape(self.Q_inferred.numpy(), self.Q_inferred.shape[:-1])
                        ])

        self.data.set_r(np.reshape(self.R.numpy(), [self.R.shape[0], self.R.shape[-1]]))

        self.data.set_d(self.D.numpy())
