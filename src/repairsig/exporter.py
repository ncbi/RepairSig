import os,sys

import pandas as pd
import numpy as np

#get logger instance
import logging
_logger = logging.getLogger(__name__)

class Exporter(object):

    def __init__(self, data):

        _logger.debug(f"Creating instance of Optimizer")

        self.data = data
        self.args = data.args
        self.outfolder = os.path.abspath(self.args.out)

        _logger.debug(f"Output folder is {self.outfolder}")

        # take care of the output folder
        try:
            os.makedirs(self.outfolder)
        except OSError as e:
            _logger.debug(f"Output folder {self.outfolder} exists.")




    def write_tables(self):
        """
        Writes the optimized tensors to file
        """

        _logger.info(f"Writing matrices to file")

        if self.data.P['fixed'] is not None:
            self.data.P['fixed'].to_csv(os.path.join(self.outfolder,'P_fixed.txt'),
                                        sep =self.args.delimiter,
                                        index=self.args.labels,
                                        header=self.args.labels)

        if self.data.P['inferred'] is not None:
            self.data.P['inferred'].to_csv( os.path.join(self.outfolder,'P_inferred.txt'),
                                            sep =self.args.delimiter,
                                            index=self.args.labels,
                                            header=self.args.labels)

        self.data.W.to_csv( os.path.join(self.outfolder,'W.txt'),
                                        sep =self.args.delimiter,
                                        index=self.args.labels,
                                        header=self.args.labels)

        self.data.A.to_csv( os.path.join(self.outfolder,'A.txt'),
                                        sep =self.args.delimiter,
                                        index=self.args.labels,
                                        header=self.args.labels)

        if self.data.Q['fixed'] is not None:
            self.data.Q['fixed'].to_csv(os.path.join(self.outfolder,'Q_fixed.txt'),
                                        sep =self.args.delimiter,
                                        index=self.args.labels,
                                        header=self.args.labels)

        if self.data.Q['inferred'] is not None:
            self.data.Q['inferred'].to_csv( os.path.join(self.outfolder,'Q_inferred.txt'),
                                            sep =self.args.delimiter,
                                            index=self.args.labels,
                                            header=self.args.labels)

        self.data.R.to_csv( os.path.join(self.outfolder,'R.txt'),
                                        sep =self.args.delimiter,
                                        index=self.args.labels,
                                        header=self.args.labels)

        self.data.D.to_csv( os.path.join(self.outfolder,'D.txt'),
                                        sep =self.args.delimiter,
                                        index=self.args.labels,
                                        header=self.args.labels)
