import os
import numpy as np
import logging
import argparse
import time
import pandas as pd
import data_utils as du
from sklearn.model_selection import train_test_split
import torch
from model import SweetWater

def main():

    parser = argparse.ArgumentParser() 
    parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=3,
                                           help= "Verbosity (between 1-4 occurrences with more leading to more "
                                                 "verbose logging). CRITICAL=0, ERROR=1, WARN=2, INFO=3, "
                                                 "DEBUG=4")
    
    parser.add_argument("-sc", "--scrna", help="scrna matrix path", default = './data/scrna_reduced_3000.tsv', type=str)
    parser.add_argument("-bulk", "--bulkrna", help="bulkrna matrix path", default = './data/bulkrna_reduced_3000.tsv', type=str)
    parser.add_argument("-dname", "--datasetname", help="name of the dataset", default= 'example', type=str)
    parser.add_argument("-n", "--nsamples", help="number of pseudobulk samples to generate", default = 5000, type=int)
    parser.add_argument("-bs", "--batchsize", help="batch size to train with", default = 256, type=int)
    parser.add_argument("-o", "--output", help="output path", default = './data/output/', type=str)

    args = parser.parse_args()
    log_levels = {
        0: logging.CRITICAL,
        1: logging.ERROR,
        2: logging.WARN,
        3: logging.INFO,
        4: logging.DEBUG,
    }
    # set the logging info
    level= log_levels[args.verbosity]
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=fmt, level=level)
    logger = logging.getLogger(__name__)
    logger.info("Starting the dataset generation process.")

    """ generate dataset """

    #load data
    logger.info("Loading data.")
    OUTPUT_PATH, datasetname, nsamples, train_size, test_size = args.output, args.datasetname, args.nsamples, 0.8, 0.2
    scRNA = pd.read_csv(args.scrna,sep='\t',index_col=0) 
    bulkRNA = pd.read_csv(args.bulkrna,sep='\t',index_col=0)
    batch_size = args.batchsize

    #split into train and test
    logger.info("Splitting data into train and test sets.")
    scRNA_train, scRNA_test = train_test_split(scRNA.copy(), stratify=scRNA.index, test_size = test_size, random_state= 42)

    # create pseudobulk for train
    logger.info("Creating pseudobulk for train and test sets.")
    xtrain, ytrain, celltypes = du.generate_synthethic(scRNA_train, nsamples = nsamples * train_size)
    xtest, ytest, _ = du.generate_synthethic(scRNA_test, nsamples = nsamples * test_size)

    ## transform and normalize
    logger.info("Transforming and normalizing data.")
    xtrain, xtest, xbulk = du.transform_and_normalize(xtrain, xtest, bulkRNA.values)

    ## convert to torch
    logger.info("Converting data to torch tensors.")
    xtrain, ytrain, xtest, ytest, xbulk = du.convert_to_float_tensors(xtrain, ytrain, xtest, ytest, xbulk)

    logger.info("Dataset generation complete.")

    """ run Sweetwater """
    logger.info("Initializing Sweetwater object.")
    ## define epochs and init sweetwater object
    epochs = round(30000/(xtrain.shape[0]/256))
    sw = SweetWater(data = (xtrain, ytrain, xtest, ytest), 
                    bulkrna = xbulk,
                    name = datasetname, verbose = True, 
                    lr = 0.00001, batch_size = 256, epochs = epochs)

    # train
    logger.info("Starting training process.")
    sw.run()
    logger.info("Training complete.")

    ## save model
    model_path = os.path.join(OUTPUT_PATH, f'{datasetname}_{batch_size}_bs_Sweetwater.pt')
    logger.info(f"Saving model to {model_path}.")
    torch.save(sw.aemodel.state_dict(), model_path)

    ## we can now infer the cell type proportions of our bulkRNA samples
    logger.info("Inferring cell type proportions of bulkRNA samples.")
    ypredbulkrna = sw.aemodel(xbulk.to(sw.device), mode = 'phase3')
    ypredbulkrna_df = pd.DataFrame(ypredbulkrna.detach().cpu(), columns = celltypes)
    logger.info("Saving deconvolved bulkRNA matrix")
    ypredbulkrna_df.to_csv(os.path.join(OUTPUT_PATH, f'{datasetname}_bulkrna.tsv'),sep='\t')
    logger.info("Inference completed")


if __name__ == '__main__':
    main()