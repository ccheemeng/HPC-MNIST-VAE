# HPC-MNIST-VAE  

This repository provides a basic implementation of a variational autoencoder (VAE) on 
the MNIST dataset with training on high-performance computing (HPC) systems 
(specifically NUS HPC) in mind.  

It is intended to serve as a reference for setting up and training more complex 
deep learning architectures on a variety of data; the MNIST dataset has thus 
been extracted into individual image files and an annotation file to closely 
simulate the structure of custom datasets.  

## Environment  

[```./environment-cpu.yml```](./environment-cpu.yml) and 
[```./environment-cuda.yml```](./environment-cuda.yml) have been provided for 
use on local machines, but note that the conda environments have been created 
for Python 3.8.5 and PyTorch 2.0.0, which the target Singularity image on NUS 
HPC uses.  

Note that [```./requirements.txt```](./requirements.txt) is intended for use in 
the HPC system.    

The following commands should replicate a working environment for the desired 
Python and PyTorch versions:  
> ```conda create -n <environment-name> python=<version>```  
> ```conda install ipykernel```  
> Desired PyTorch installation (see 
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/))  
> ```pip install matplotlib pandas tqdm```  

## Data  

The original MNIST data files are provided in [```./data/```](./data/). Run all 
cells in [```./extract-mnist.ipynb```](./extract-mnist.ipynb) to extract the 
individual image files and create ```.csv``` annotations.  

Move the ```./train/``` and ```./test/``` directories into 
[```./data/```](./data/).  

## Training  

```scp``` the following files and directories into the target working directory 
on the HPC system:  
* Data files  
* [```./modules/```](./modules/)  
* [```./utils/```](./utils/)  
* [```./requirements.txt```](./requirements.txt)  
* [```./train.pbs```](./train.pbs)
* [```./train.py```](./train.py)  

Set up the necessary packages in the desired Singularity image:  
> ```module load singularity```  
> ```singularity exec <singularity-image> bash```  
> ```pip install -r requirements.txt```  
> ```exit```  

Ensure that [```./train.pbs```](./train.pbs) will load the desired Singularity 
image. Modify training hyperparameters and PBS requested compute if necessary.  

Submit the job to the queue:  
> ```qsub train.pbs```  

Check the status of the job (Q: queue, R: running, E: error, F: finished):  
> ```qstat -xfn```  

```stderr.$PBS_JOBID``` will periodically update to reflect console outputs of 
```train.py```.  

When the job is complete, the state dictionaries of the VAE and the Adam 
optimiser will be saved in the working directory, together with a log of 
hyperparameters and training loss. These files can be ```scp``` back to the 
local machine for use.  
