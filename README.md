# Sweetwater
An interpretable and adaptive autoencoder for efficient tissue deconvolution. 
This paper is available in bioRxiv (https://arxiv.org/pdf/2311.11991) and is currently under review.

## Abstract
**Motivation**:
Single-cell RNA-sequencing (scRNA-seq) stands as a powerful tool for deciphering cellular heterogeneity and exploring
gene expression profiles at high resolution. However, its high cost renders it impractical for extensive sample cohorts within
routine clinical care, hindering its broader applicability. Hence, many methodologies have recently arised to estimate cell
type proportions from bulk RNA-seq samples (known as deconvolution methods). However, they have several limitations:

* Many depend on selecting a robust scRNA-seq reference dataset, which is often challenging. 
* Secondly, building reliable pseudobulk samples requires determining the optimal number of genes or cells involved in the simulated data generation
process, which has not been studied in depth. Moreover, pseudobulk and bulk RNA-seq samples often exhibit distribution
shifts. 
* Finally, most modern deconvolution approaches behave as a black box, and the underlying mechanisms of the
deconvolution task are still unknown, which can compromise the reliability of the results.

**Results**:
In this work, we present Sweetwater, an adaptive and interpretable autoencoder able to efficiently deconvolve bulk
RNA-seq and microarray samples leveraging multiple classes of reference data, such as scRNA-seq and single-nuclei
RNA-seq. Moreover, it can be trained on a mixture of FACS-sorted FASTQ files, which we newly propose to use as this
reduces platform-specific biases and may potentially outperform single-cell-based references. Also, we demonstrate that
Sweetwater effectively uncovers biologically meaningful patterns during the training process, increasing the reliability of
the results.

<p align="center" width="100%">
    <img width="100%" src="https://raw.githubusercontent.com/ubioinformat/Sweetwater/main/imgs/figure1_nature.png">
</p>


## Summary of Sweetwater Architecture

Encoder:

| Parameter type          | Value                       |
|-------------------------|-----------------------------|
| Layer type              | Linear                      |
| Number of hidden layers | 2                           |
| Activation function     | ReLU                        |
| Embedding dimension     | $$ G \rightarrow G^2 \rightarrow G^4 $$ |

Decoder: 

| Parameter type            | Value           |
|---------------------------|-----------------|
| Layer type                | Linear          |
| Number of hidden layer    | 2               |
| Activation function       | ReLU            |
| Embedding dimension       | \( G^4 → G^2 → G \) |

Deconvolver:

| Parameter type            | Value    |
|---------------------------|----------|
| Layer type                | Linear   |
| Number of hidden layer    | 2        |
| Activation function       | ReLU, Softmax    |
| Embedding dimension       | \( G^4 → G^2 → C_t \) |

The model is trained with:

| Parameter type            | Value         |
|---------------------------|---------------|
| Optimizer                 | Adam          |
| Learning rate             | 0.00001       |
| Batch size                | 256           |
| EarlyStopper Phase I      | patience = 10 |
| EarlyStopper Phase II     | patience = 10 |
| EarlyStopper Phase III    | patience = 50 |

Here \( G \) is the number of input genes and \( C_t \) is the number of cell types to be deconvolved. 

## Build docker 

Build image as (in the same folder as the Dockerfile):
```
docker build -t <image_name> docker/.
# docker build -t sweetwater .
```

To run the container
```
docker run -dt --gpus all --name <container_name> <image_name>
```

-v flag may be added to work in the same folder (-v your_path/Sweetwater/:/wdir/)



## Run the model 

**Parameters:**

### -sc, --scrna
- **Description**: Path to the single-cell RNA matrix file. Should have annotated cell types as the index and genes as the columns.
- **Default Value**: `data/scrna_reduced_3000.tsv`
- **Type**: `str`

### -bulk, --bulkrna
- **Description**: Path to the bulk RNA matrix file. Should have genes as the columns, in the same order as the scRNA reference file.
- **Default Value**: `data/bulkrna_reduced_3000.tsv`
- **Type**: `str`

### -dname, --datasetname
- **Description**: Name of the dataset.
- **Default Value**: `example`
- **Type**: `str`

### -n, --nsamples
- **Description**: Number of pseudobulk samples to generate. Default is recommended.
- **Default Value**: 5000
- **Type**: int

### -bs, --batchsize
- **Description**: Batch size to train with. Default is recommended.
- **Default Value**: 256
- **Type**: int

### -o, --output
- **Description**: Output path for saving the trained model and deconvolved bulkRNA dataframe.
- **Default Value**: `data/output/`
- **Type**: `str`

**Usage:** To run Sweetwater you can use the default parameters, which will use a reduced human brain cortex dataset provided in the examples folder. 
Make sure the *scrna_reduced_3000.tsv* matrix is available in the data folder by **unziping the scrna_reduced_3000.zip** file. This contains a reduced 
version of a human brain cortex dataset with only the top 3000 most variant genes

```
python3 src/main.py
```

Additionally, we have included a *deconvolution.ipynb* file where it is showed how to deconvolve an expression matrix using a scRNA-seq reference and  
an *interpretability.ipynb* file that allow to perform the interpretability analysis showed in the manuscript. Again, it is necessary to  
to **unzip the scrna_reduced_3000.zip** matrix first before running these. For the *fastq_generation_part1.ipynb* and *fastq_generation_part2.ipynb* scripts, a group of empty files 
have been provided as a placeholder.


