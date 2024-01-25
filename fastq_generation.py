import pandas as pd
import os
import numpy as np
import subprocess
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict
import itertools
import random

## list dirs in fastq
fpath_fq = os.path.join('data','aml_mds','healthy_young_sorted','fastq')
fpath_fqd = os.path.join('data','aml_mds','healthy_young_sorted','fastq_decompressed_isolated')
fpath_fqdm = os.path.join('data','aml_mds','healthy_young_sorted','fastq_decompressed_mixed')
fpath_index = os.path.join('data','aml_mds','healthy_young_sorted','homo_sapiens_ref','homo_sapiens','transcriptome.idx')
fastq_l = os.listdir(fpath_fq)

"""
This is for isolated fastqs
"""

def generate_isolated_fastqs():
    
    #read fastq we are using
    ref = pd.read_csv(os.path.join('data','aml_mds','healthy_young_sorted','lists_of_fastq.txt'),header=None).iloc[:,0].values

    for fq in tqdm(fastq_l, desc= 'Decompressing fastqs...'):
        name = fq.split('.')[0]
        if name in ref:

            if os.path.exists(os.path.join(fpath_fqd,f'{name}_counts')):
                continue

            #decompress the file and send it to fastq_decompressed
            gzip = f"gunzip -c {fpath_fq}/{name}.fastq.gz > {fpath_fqd}/{name}.fastq"
            subprocess.call(gzip, shell=True)

            #now quantify
            kallisto_quant = f'kallisto quant -i {fpath_index} -o {fpath_fqd}/{name}_counts --single -l 200 -s 20 -t 64 {fpath_fqd}/{name}.fastq'
            subprocess.call(kallisto_quant, shell=True)


"""
This is for mixed fastqs
"""

##now, mixed it
def generate_mixed_fastqs(list_of_fastqs, nsamples=1000, mode='train'):

    def combine_fastqs(sample_props, fqd):

        def read_fastq_seqs(filepath, prop, ct):

            lines_list = []
            with open(filepath, 'r') as fh:
                for seq_header, seq, qual_header, qual in itertools.zip_longest(*[fh] * 4):
                    
                    if any(line is None for line in (seq_header, seq, qual_header, qual)):
                        raise Exception(
                            "Number of lines in FASTQ file must be multiple of four "
                            "(i.e., each record must be exactly four lines long).")
                    if not seq_header.startswith('@'):
                        raise Exception("Invalid FASTQ sequence header: %r" % seq_header)
                    if qual_header != '+\n':
                        raise Exception("Invalid FASTQ quality header: %r" % qual_header)
                    if qual == '\n':
                        raise Exception("FASTQ record is missing quality scores.")

                    lines_list.append("".join([seq_header, seq, qual_header, qual]))

            #randomly select lines according to proportions
            if prop < 1:
                n = int(prop * len(lines_list))
                subsampled_lines_list = random.sample(lines_list, n)
                print(f"{n} lines sampled from fastq, from {prop} proportions of {ct} celltype")
            else:
                print(f"all lines sampled from fastq, from proportions {ct}")

            return subsampled_lines_list

        def select_fastq_piece(name, prop, ct):
            fastq_lines = read_fastq_seqs(os.path.join(fpath_fqd,f'{name}.fastq'), prop, ct)
            return fastq_lines

        mixedfastq = []
        names = []
        for ct, prop in tqdm(sample_props, 'mixing fastq according to proportions'): #iterate celltype-proportion tuplas

            if prop > 0:
                name = random.sample(fqd[ct],1)[0]
                names.append(name)
                mixedfastq += select_fastq_piece(name, prop, ct)

        #write fastq
        #generate a random id
        rid = ''.join(map(str,random.sample(range(1000),4)))

        #write the rid-names map
        with open(os.path.join(fpath_fqdm,mode,f'mixed_{mode}_names_list.csv'), 'a') as file:
            file.write(','.join([rid, '||'.join(names)]) + '\n')

        #retrieve proportions
        proportions_str = list(map(lambda x: str(x[1]), sample_props))

        with open(os.path.join(fpath_fqdm,mode,f'mixed_{mode}_proportions.csv'), 'a') as file:
            file.write(','.join(proportions_str) + '\n')

        ## now save the fastq
        with open(os.path.join(fpath_fqdm,mode,'samples',f'mixed_{mode}_sample_id{rid}.fastq'), 'w') as file:
            for line in mixedfastq:
                file.write(line)

        #now quantify (generate the count matrix for the fastq)
        kallisto_quant = f'kallisto quant -i {fpath_index} -o {os.path.join(fpath_fqdm,mode,"samples")}/mixed_{mode}_sample_id{rid}_counts --single -l 200 -s 20 -t 64 {os.path.join(fpath_fqdm,mode,"samples",f"mixed_{mode}_sample_id{rid}.fastq")}'
        subprocess.call(kallisto_quant, shell=True)

        ## now remove the fastq file
        os.remove(os.path.join(fpath_fqdm,mode,'samples',f'mixed_{mode}_sample_id{rid}.fastq'))

    #define celltypes
    celltypes = ["GMP", "HSC", "MEP"]

    ## generate dict of fastqs according to the celltype
    fqd = defaultdict(list)
    for f in list_of_fastqs:
        for ct in celltypes:
            if ct in f:
                fqd[ct].append(f)

    nct = len(celltypes)

    ## init empty list to fill it with generated combinations
    combl = []
    s = 1
    for i in range(1, nct+1): ## group of combinations
        if i > 1:
            s = 10
        cmb = list(combinations(range(nct),i))
        for e in cmb:
            combelm = np.zeros((s, nct))
            combelm[:,e] = np.random.dirichlet(alpha = np.ones(len(e)), size = s)
            combl.append(combelm)

    ## generate benchmarking proportions
    benchmark_props = pd.DataFrame(np.vstack(combl))
    benchmark_props.columns = celltypes

    ## go through the matrix, until n samples have been generated
    n = 0
    while n < nsamples:

        if not (n%5):
            print(f"number of samples {n}")

        for i in range(benchmark_props.shape[0]):

            sample_props = list(zip(celltypes, benchmark_props.iloc[i,:]))
            combine_fastqs(sample_props, fqd)
            n+=1


#first, retrieve the files from the isolated folder
files = []
for item in os.listdir(fpath_fqd):
    item_path = os.path.join(fpath_fqd, item)
    if os.path.isfile(item_path):
        files.append(item.split('.')[0])

#now, separate train and test
train_test_ref = pd.read_csv(os.path.join('data', 'aml_mds', 'healthy_young_sorted','pseudobulk','train_test_samples.txt'),header=None)
train_files_ref = train_test_ref[train_test_ref.iloc[:,1] == 'train'].iloc[:,0].values
train_files = [f for f in files if f in train_files_ref]
test_files = [f for f in files if f not in train_files_ref]

## first for train
generate_mixed_fastqs(train_files, nsamples = 200, mode ='train')

