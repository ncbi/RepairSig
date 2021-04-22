import os,sys
import pandas as pd
import numpy as np

#get logger instance
import logging
_logger = logging.getLogger(__name__)

class Resources(object):


    def __init__(self,args):
        """ Initializes all tensors by either reading content from file, or
        creating random initial data for it.

        """
        _logger.debug(f"Creating Resources instance.")

        self.args = args

        # We Store the tensors as named pandas dataframes internally so we can export them later
        # Numpy arrays do not natively support row names.

        # format for M
        # {i:['dim_id', df]} where i corresponds to the index of the first dimension
        # in M, dim_id is the filename and df the pandas dataframe
        # obtained with initialize_m
        self.M = None

        # format for P and Q
        # since P can be a mixture of fixed and unknown signatures,
        # we need to account for it by splitting it into these two
        # components. Hence {'fixed':[None or df], 'inferred':[None or df]}
        self.P = None
        self.Q = None

        self.A = None
        self.W = None
        self.R = None
        self.D = None

        _logger.info(f"Initializing tensors.")
        self.initizalize_m()
        self.initizalize_p()
        self.initizalize_w()
        self.initizalize_a()
        self.initizalize_q()
        self.initizalize_r()
        self.initizalize_d()
        _logger.info(f"Tensor initializtion complete.")


        _logger.info(f"Final tensor shapes are [dimensions with a \"+\" refer to the fixed (left) and the to be inferred component (right)]:")
        _logger.info(f"M: [{self.M[0][1].shape[0]}, {self.M[0][1].shape[1]}, {len(self.M)}]")
        _logger.info(f"P: [{0 if self.P['fixed'] is None else self.P['fixed'].shape[0]} + {0 if self.P['inferred'] is None else self.P['inferred'].shape[0]},  {self.P['fixed'].shape[1] if self.P['fixed'] is not None else self.P['inferred'].shape[1]}]")
        _logger.info(f"W: [{self.W.shape[0]}, {self.W.shape[1]}]")
        _logger.info(f"A: [{self.A.shape[0]}, {self.A.shape[1]}]")
        _logger.info(f"Q: [{0 if self.Q['fixed'] is None else self.Q['fixed'].shape[0]} + {0 if self.Q['inferred'] is None else self.Q['inferred'].shape[0]},  {self.Q['fixed'].shape[1] if self.Q['fixed'] is not None else self.Q['inferred'].shape[1]}]")
        _logger.info(f"R: [{self.R.shape[0]}, {self.R.shape[1]}]")
        _logger.info(f"D: [{self.D.shape[0]}, {self.D.shape[1]}]")


    def initizalize_m(self):
        """Depending on the parameters specified by the user, load M from file(s)
        """

        _logger.debug(f"Starting reading M")

        # First, make sure all the files specified for M exist on volatile memory
        files = [os.path.abspath(x) for x in self.args.matrix]

        for f in files:
            if not os.path.exists(f):
                _logger.critical(f"File '{os.path.basename(f)}' for input matrix M could not be found. Exiting.")
                sys.exit(1)

        _logger.debug(f"Found {len(files)} files on disk. Parsing.")

        # Next, read in the files one by one, in the order as specified by the user
        self.M = {}
        for i,f in enumerate(files):
            m = pd.read_csv(f,
                            sep = self.args.delimiter,
                            header = 0 if self.args.labels else None,     # Take care of the labels flag
                            index_col = 0 if self.args.labels else False #
                )

            # If no labels were specified by the user, add generic ones
            if not self.args.labels:
                m.columns = [f"mut_cat_{x+1}" for x in range(m.shape[1])]
                m.index = [f"genome_{x+1}" for x in range(m.shape[0])]

            # Cast to float
            m = m.astype(np.float64, copy=False, errors='raise')

            _logger.debug(m)

            _logger.debug(f"Got M_{i} with shape {m.shape}")

            self.M[i] = [os.path.basename(f), m]

        # Sanity check. All dataframes must have the same dimensions
        if len(set([m[1].shape[0] for m in self.M.values()])) != 1 or len(set([m[1].shape[0] for m in self.M.values()])) != 1:
            _logger.critical(f"The input matrices for M must be of equal dimension. Current dimensions are:")
            for i in range(len(self.M)):
                _logger.critical(f"{self.M[i][0]} has shape [{self.M[i][1].shape[0]},{self.M[i][1].shape[1]}]")
            _logger.critical(f"Exiting.")
            sys.exit(1)

        _logger.info(f"Successfully read M with shape [{len(self.M)},{self.M[0][1].shape[0]},{self.M[0][1].shape[1]}]")

    def initizalize_p(self):
        """Depending on the parameters specified by the user, load P from file(s),
        and/or initialize random matrix according to NUMPRI

        At the end of this fuction, either both or at least one of the options for
        Q will be present.

        Args:
          args: parser arguments
        """

        _logger.debug(f"Starting initizalize_p")

        # we need to enforce that M is read from file first, as the dimension
        # compatibility depends on it
        if self.M is None:
            _logger.debug(f"initialize_m has not been called yet. Make sure you call this function first. ")
            _logger.critical(f"Internal error (no M). Exiting.")
            sys.exit(1)

        self.P = {} # initialize

        if self.args.primary is not None: # we have a data for static primary features

            # make sure the file exists on disk
            file = os.path.abspath(self.args.primary)
            if not os.path.exists(file):
                _logger.critical(f"File '{os.path.basename(file)}' for input matrix P could not be found. Exiting.")
                sys.exit(1)

            # read in the data
            p = pd.read_csv(file,
                            sep = self.args.delimiter,
                            header = 0 if self.args.labels else None,    # Take care of the labels flag
                            index_col = 0 if self.args.labels else False #
                )

            # cast to float
            p = p.astype(np.float64, copy=False, errors='raise')

            # If no labels were specified by the user, add generic ones
            if not self.args.labels:
                p.columns = [f"mut_cat_{x+1}" for x in range(p.shape[1])]
                p.index = [f"fixed_sig_{x+1}" for x in range(p.shape[0])]

            # Sanity check: the number of mutational categories must be identical to the one in M
            if self.M[0][1].shape[1] != p.shape[1]:
                _logger.critical(f"The number of mutational signatures of M and P must be identical. Currently, M has {self.M[0][1].shape[0]}, and P has {p.shape[1]} signatures. Exiting.")
                sys.exit(1)

            # add to dictionary
            self.P['fixed'] = p

            _logger.info(f"Successfully read P with shape [{p.shape[0]},{p.shape[1]}]")

        else: # no primary specified
            self.P['fixed'] = None

        # next, check if we have primary signatures to infer
        if self.args.numpri is not None:
            _logger.debug(f"Case N is not None")

            # initialize with random data
            p = pd.DataFrame(   np.random.rand(self.args.numpri, self.M[0][1].shape[1]), # since the dimension of p in this case depends on knowing the number of categories, we source it from M
                                columns=self.M[0][1].columns,
                                index=[f"RepairSig_p{x+1}" for x in range(self.args.numpri)]
                            )

            # cast to float
            p = p.astype(np.float64, copy=False, errors='raise')

            # add to dictionary
            self.P['inferred'] = p

            _logger.debug(f"Successfully created {self.args.numpri} primary signatures to be infered with shape [{p.shape[0]},{p.shape[1]}]")

        else: # no N
            self.P['inferred'] = None

        #Final report
        p_dim1 = (0 if self.P['fixed'] is None else self.P['fixed'].shape[0]) + (0 if self.P['inferred'] is None else self.P['inferred'].shape[0])
        p_dim2 = self.P['fixed'].shape[1] if self.P['fixed'] is not None else self.P['inferred'].shape[1]
        _logger.debug(f"Final shape for P is [{p_dim1},{p_dim2}]")


    def initizalize_w(self):
        """Initializes W

        Args:
          args: parser arguments
        """

        _logger.debug(f"Starting initizalize_w")

        # Sanity check. We need M and P to be present in order to determine the dimension of W
        if self.M is None or self.P is None:
            _logger.debug(f"initialize_m or initialize_p has not been called yet. Make sure you call this function first.")
            _logger.critical(f"Internal error (no M or P).  Exiting.")
            sys.exit(1)

        # determine the number of primary signatures and their row names from P
        num_pri_sigs = (0 if self.P['fixed'] is None else self.P['fixed'].shape[0]) + (0 if self.P['inferred'] is None else self.P['inferred'].shape[0])
        pri_sigs_index = ([] if self.P['fixed'] is None else list(self.P['fixed'].index)) + ([] if self.P['inferred'] is None else list(self.P['inferred'].index))


        # initialize with random data
        w = pd.DataFrame(   np.random.rand(num_pri_sigs, len(self.M)), # since the dimension of p in this case depends on knowing the number of categories, we source it from M
                            columns=[f"{self.M[x][0]}" for x in range(len(self.M))],
                            index=pri_sigs_index
                        )

        # cast to float
        w = w.astype(np.float64, copy=False, errors='raise')

        # add to dictionary
        self.W = w

        _logger.debug(f"Successfully created W with shape [{w.shape[0]},{w.shape[1]}]")


    def initizalize_a(self):
        """Initializes A

        Args:
          args: parser arguments
        """

        _logger.debug(f"Starting initizalize_a")

        # Sanity check. We need M and P to be present in order to determine the dimension of W
        if self.M is None or self.P is None:
            _logger.debug(f"initialize_m or initialize_p has not been called yet. Make sure you call this function first.")
            _logger.critical(f"Internal error (no M or P).  Exiting.")
            sys.exit(1)

        # determine the number of primary signatures and their row names from P
        num_pri_sigs = (0 if self.P['fixed'] is None else self.P['fixed'].shape[0]) + (0 if self.P['inferred'] is None else self.P['inferred'].shape[0])
        pri_sigs_columns = ([] if self.P['fixed'] is None else list(self.P['fixed'].index)) + ([] if self.P['inferred'] is None else list(self.P['inferred'].index))

        # initialize with random data
        a = pd.DataFrame(   np.random.rand(self.M[0][1].shape[0], num_pri_sigs),
                            index=self.M[0][1].index,
                            columns=pri_sigs_columns
                        )

        # cast to float
        a = a.astype(np.float64, copy=False, errors='raise')

        # add to dictionary
        self.A = a

        _logger.debug(f"Successfully created A with shape [{a.shape[0]},{a.shape[1]}]")


    def initizalize_q(self):
        """Depending on the parameters specified by the user, load Q from file(s),
        and/or initialize random matrix according to NUMSEC

        At the end of this fuction, either both or at least one of the options for
        P will be present.

        Args:
          args: parser arguments
        """

        _logger.debug(f"Starting initizalize_q")

        # we need to enforce that M is read from file first, as the dimension
        # compatibility depends on it
        if self.M is None:
            _logger.debug(f"initialize_m has not been called yet. Make sure you call this function first. ")
            _logger.critical(f"Internal error (no M). Exiting.")
            sys.exit(1)

        self.Q = {} # initialize

        if self.args.secondary is not None: # we have a data for static secondary features

            # make sure the file exists on disk
            file = os.path.abspath(self.args.secondary)
            if not os.path.exists(file):
                _logger.critical(f"File '{os.path.basename(file)}' for input matrix Q could not be found. Exiting.")
                sys.exit(1)

            # read in the data
            q = pd.read_csv(file,
                            sep = self.args.delimiter,
                            header = 0 if self.args.labels else None,    # Take care of the labels flag
                            index_col = 0 if self.args.labels else False #
                )

            # If no labels were specified by the user, add generic ones
            if not self.args.labels:
                q.columns = [f"mut_cat_{x+1}" for x in range(q.shape[1])]
                q.index = [f"fixed_sig_{x+1}" for x in range(q.shape[0])]

            # cast to float
            q = q.astype(np.float64, copy=False, errors='raise')

            # Sanity check: the number of mutational categories must be identical to the one in M
            if self.M[0][1].shape[1] != q.shape[1]:
                _logger.critical(f"The number of mutational signatures of M and Q must be identical. Currently, M has {self.M[0][1].shape[0]}, and Q has {q.shape[1]} signatures. Exiting.")
                sys.exit(1)

            # add to dictionary
            self.Q['fixed'] = q

            _logger.info(f"Successfully read Q with shape [{q.shape[0]},{q.shape[1]}]")

        else: # no primary specified
            self.Q['fixed'] = None

        # next, check if we have primary signatures to infer
        if self.args.numsec is not None:
            _logger.debug(f"Case J is not None")

            # initialize with random data
            q = pd.DataFrame(   np.random.rand(self.args.numsec, self.M[0][1].shape[1]), # since the dimension of q in this case depends on knowing the number of categories, we source it from M
                                columns=self.M[0][1].columns,
                                index=[f"RepairSig_p{x+1}" for x in range(self.args.numsec)]
                            )

            # cast to float
            q = q.astype(np.float64, copy=False, errors='raise')

            # add to dictionary
            self.Q['inferred'] = q

            _logger.debug(f"Successfully created {self.args.numsec} secondary signatures to be infered with shape [{q.shape[0]},{q.shape[1]}]")

        else: # no N
            self.Q['inferred'] = None

        #Final report
        q_dim1 = (0 if self.Q['fixed'] is None else self.Q['fixed'].shape[0]) + (0 if self.Q['inferred'] is None else self.Q['inferred'].shape[0])
        q_dim2 = self.Q['fixed'].shape[1] if self.Q['fixed'] is not None else self.Q['inferred'].shape[1]
        _logger.debug(f"Final shape for Q is [{q_dim1},{q_dim2}]")


    def initizalize_r(self):
        """Initializes R

        Args:
          args: parser arguments
        """

        _logger.debug(f"Starting initizalize_r")

        # Sanity check. We need M and Q to be present in order to determine the dimension of R
        if self.M is None or self.Q is None:
            _logger.debug(f"initialize_m or initialize_q has not been called yet. Make sure you call this function first.")
            _logger.critical(f"Internal error (no M or Q).  Exiting.")
            sys.exit(1)

        # determine the number of secondary signatures and their names from Q
        num_sec_sigs = (0 if self.Q['fixed'] is None else self.Q['fixed'].shape[0]) + (0 if self.Q['inferred'] is None else self.Q['inferred'].shape[0])
        sec_sigs_index = ([] if self.Q['fixed'] is None else list(self.Q['fixed'].index)) + ([] if self.Q['inferred'] is None else list(self.Q['inferred'].index))
        col_names = [self.M[x][0] for x in range(len(self.M))]

        # initialize with random data
        r = pd.DataFrame(   np.random.rand(num_sec_sigs, len(self.M)), # since the dimension of p in this case depends on knowing the number of categories, we source it from M
                            columns=col_names,
                            index=sec_sigs_index
                        )

        # cast to float
        r = r.astype(np.float64, copy=False, errors='raise')

        # add to dictionary
        self.R = r

        _logger.debug(f"Successfully created R with shape [{r.shape[0]},{r.shape[1]}]")


    def initizalize_d(self):
        """Initializes D

        Args:
          args: parser arguments
        """

        _logger.debug(f"Starting initizalize_d")

        # Sanity check. We need M and Q to be present in order to determine the dimension of R
        if self.M is None or self.Q is None:
            _logger.debug(f"initialize_m or initialize_q has not been called yet. Make sure you call this function first.")
            _logger.critical(f"Internal error (no M or Q).  Exiting.")
            sys.exit(1)

        # determine the number of secondary signatures and their names from Q
        num_sec_sigs = (0 if self.Q['fixed'] is None else self.Q['fixed'].shape[0]) + (0 if self.Q['inferred'] is None else self.Q['inferred'].shape[0])
        col_names = ([] if self.Q['fixed'] is None else list(self.Q['fixed'].index)) + ([] if self.Q['inferred'] is None else list(self.Q['inferred'].index))

        # initialize with random data
        d = pd.DataFrame(   np.random.rand(self.M[0][1].shape[0], num_sec_sigs),
                            index=self.M[0][1].index,
                            columns=col_names
                        )

        # cast to float
        d = d.astype(np.float64, copy=False, errors='raise')

        # add to dictionary
        self.D = d

        _logger.debug(f"Successfully created D with shape [{d.shape[0]},{d.shape[1]}]")


    def _get_dynamic_tensor(self, t, t_name="tensor"):
        """ Generic getter function for P and Q.
        Returns an array of numpy objects ['fixed', 'inferred'] with the content of t

            Args:
                t: The 2-dimensional tensor to return
                t_name: string representing the name of t for output purposes

            Returns:
                array with [ 'fixed' or None, 'inferred' or None]
        """
        _logger.debug(f"_get_dynamic_tensor() for {t_name}")

        # Sanity check. Has P been initialized?
        if t is None:
            _logger.critical(f"Tensor {t_name} has not yet been initialized. Call the corresponding initialize function first. Exiting")
            sys.exit(1)

        return [ None if t['fixed'] is None else t['fixed'].to_numpy(copy=True), None if t['inferred'] is None else t['inferred'].to_numpy(copy=True)]


    def _set_dynamic_tensor(self, T, t, t_name="tensor"):
        """ Generic setter for P and Q.
        Sets the content of m to the corresponding pandas tables in M

        Args:
            T: Pandas dataframe to be set as t
            t: The 2-dimensional tensor to set
            t_name: string representing the name of t for output purposes

        """
        _logger.debug(f"_set_dynamic_tensor() for {t_name}")

        # Sanity check. Has M been initialized?
        if T is None:
            _logger.critical(f"Tensor {t_name} has not yet been initialized. Call initialize first. Exiting")
            sys.exit(1)

        # Sanity check. Are the dimensions of p the same as P?
        if T['fixed'] is None and t[0] is not None:
            _logger.critical(f"Trying to set {t_name} with a shape different from the one present.")
            _logger.debug(f"Passed shape for fixed component is {list(t[0].shape)} whereas stored shape is None. Exiting.")
            sys.exit(1)

        if T['fixed'] is not None and T['fixed'].shape != t[0].shape:
            _logger.critical(f"Trying to set {t_name} with a shape different from the one present.")
            _logger.debug(f"Passed shape for fixed component is {list(t[0].shape)} whereas stored shape is {T['fixed'].shape}. Exiting.")
            sys.exit(1)

        if T['inferred'] is None and t[1] is not None:
            _logger.critical(f"Trying to set {t_name} with a shape different from the one present.")
            _logger.debug(f"Passed shape for inferred component is {list(t[1].shape)} whereas stored shape is None. Exiting.")
            sys.exit(1)

        if T['inferred'] is not None and T['inferred'].shape != t[1].shape:
            _logger.critical(f"Trying to set {t_name} with a shape different from the one present.")
            _logger.debug(f"Passed shape for fixed component is {list(t[1].shape)} whereas stored shape is {T['inferred'].shape}. Exiting.")
            sys.exit(1)

        # Set values
        if t[0] is not None:
            T['fixed'][:] = t[0]

        if t[1] is not None:
            T['inferred'][:] = t[1]

    def _get_tensor(self, t, t_name="tensor"):
        """ Generic getter for W, A, R, and D as they all behave the same

            Args:
                t: The 2-dimensional tensor to return
                t_name: string representing the name of t for output purposes

            Returns:
                Copy of t as a numpy object
        """

        _logger.debug(f"getTensor() for {t_name}")

        # Sanity check. Has t been initialized?
        if t is None:
            _logger.critical(f"Tensor {t_name} has not yet been initialized. Call the appropriate initialize function first. Exiting")
            sys.exit(1)

        return t.to_numpy(copy=True)

    def _set_tensor(self,T, t, t_name="tensor"):
        """ Generic setter for W, A, R, and D as they all behave the same

            Args:
                T: Pandas dataframe to be set as t
                t: The 2-dimensional tensor to set
                t_name: string representing the name of t for output purposes

        """

        _logger.debug(f"setTensor() for {t_name}")

        # Sanity check. Has t been initialized?
        if t is None:
            _logger.critical(f"Tensor {t_name} has not yet been initialized. Call the appropriate initialize function first. Exiting")
            sys.exit(1)

        # Do shapes match?
        if T.shape != t.shape:
            _logger.critical(f"Trying to set {t_name} with a shape different from the one present.")
            _logger.debug(f"Passed shape is {list(t.shape)} whereas stored shape is {T.shape}. Exiting.")
            sys.exit(1)

        # Then set
        T[:] = t


    def get_m(self):
        """ Returns a numpy object with the content of M

            Returns:
                numpy with shape (GxKxL)
        """
        _logger.debug(f"get_m()")

        # Sanity check. Has M been initialized?
        if self.M is None:
            _logger.critical(f"M has not yet been initialized. Call initialize_m() first. Exiting")
            sys.exit(1)

        return np.dstack( [ self.M[x][1].to_numpy(copy=True)[:, :, np.newaxis] for x in range(len(self.M))] )


    def set_m(self, m):
        """ Sets the content of m to the corresponding pandas tables in M

        """
        _logger.debug(f"set_m()")

        # Sanity check. Has M been initialized?
        if self.M is None:
            _logger.critical(f"M has not yet been initialized. Call initialize_m() first. Exiting")
            sys.exit(1)

        # Sanity check. Are the dimensions of m the same as M?
        if m.shape[0] != self.M[0][1].shape[0] or m.shape[1] != self.M[0][1].shape[1] or m.shape[2] != len(self.M) or len(m.shape) != 3:
            _logger.critical(f"Trying to set M with a shape different from the one present.")
            _logger.debug(f"Passed shape is {list(m.shape)} whereas stored shape is [{self.M[0][1].shape[0]}, {self.M[0][1].shape[1]}, {len(self.M)}]. Exiting.")
            sys.exit(1)

        # Set values
        for x in range(len(self.M)):
            self.M[x][1][:] = m[:, :, x]



    #define convecience functions for P, W, A, R, Q, and D.
    def get_p(self):
        return self._get_dynamic_tensor(self.P, 'P')

    def set_p(self, p):
        return self._set_dynamic_tensor(self.P, p, 'P')

    def get_q(self):
        return self._get_dynamic_tensor(self.Q, 'Q')

    def set_q(self, q):
        return self._set_dynamic_tensor(self.Q, q, 'Q')

    def get_w(self):
        return self._get_tensor(self.W, 'W')

    def set_w(self, w):
        self._set_tensor(self.W,w,"W")

    def get_a(self):
        return self._get_tensor(self.A, 'A')

    def set_a(self, a):
        self._set_tensor(self.A,a,"A")

    def get_r(self):
        return self._get_tensor(self.R, 'R')

    def set_r(self, r):
        self._set_tensor(self.R,r,"R")

    def get_d(self):
        return self._get_tensor(self.D, 'D')

    def set_d(self, d):
        self._set_tensor(self.D,d,"D")
