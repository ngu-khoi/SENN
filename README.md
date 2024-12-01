# Self-Explaining Neural Networks: Testing Robustness on 3D Rotational Perturbations

This repository extends the work on Self-Explaining Neural Networks (SENN) [1], specifically focusing on testing their robustness against 3D rotational perturbations. This implementation builds upon the excellent work done in "SENN: A Review with Extensions" [3] by Hussain et al., which provided a thorough analysis and improvements to the original SENN framework.

Our extension investigates how SENN models perform when faced with 3D rotational perturbations, exploring the stability and reliability of their explanations under these transformations. We aim to understand whether the interpretability benefits of SENN are preserved when objects are viewed from different angles in 3D space.

## Table of Contents
- [Self-Explaining Neural Networks: Testing Robustness on 3D Rotational Perturbations](#self-explaining-neural-networks-testing-robustness-on-3d-rotational-perturbations)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [How to run?](#how-to-run)
  - [Results](#results)
  - [Authors](#authors)
  - [References](#references)

## Project Structure
<img src="images/UML-SENN.png" alt="Project Structure" width="720">

## How to run?

1. Clone and set up Python environment
```bash
# Clone repository
git clone https://github.com/ngu-khoi/SENN
cd SENN

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

2. To reproduce our results using trained models, run the [Report Notebook](report.ipynb).  

3. To train a model using one of our experiment parameters:
```bash
python main.py --config "configs/compas_lambda1e-4_seed555.json"
```

4. To train a new model or perform a new experiment:
```bash
python main.py --config "./config.json"
```

Where *config.json* is prepared according to the template below:

```
{
  "exp_name": "exp001",                         (str, the name of the experiment, used to save the checkpoints and csv results)
  "data_path": "datasets/data/mnist_data",      (str, the path where the data is to be saved)
  "model_class": "SENN"/"DiSENN",               (str, whether to create a SENN or a DiSENN model)
  "pretrain_epochs": 1,                         (int, the number of epochs  to pretrain a beta-VAE for)
  "pre_beta": 1.0,                              (float, the beta to be used in case of DiSENN (VAE pretraining))
  "beta": 4.0,                                  (float, the beta to be used in case of DiSENN (DiSENN training))
  "train": true/false,                          (bool, whether to train the model or not)
  "dataloader": "compas"/"mnist",               (str, the name of the dataloader to be used)
  "conceptizer": "Conceptizer",                 (str, the name of the conceptizer class to be used)
  "parameterizer": "Parameterizer",             (str, the name of the parameterizer class to be used)
  "aggregator": "Aggregator",                   (str, the name of the aggregator class to be used)
  "image_size": 28,                             (int, the size of the input images)
  "num_concepts": 5,                            (int, the number of concepts to be used in training)
  "num_classes": 10,                            (int, the number of output classes)
  "dropout": 0.5,                               (float, the dropout value to be used during training)
  "device": "cuda:0"/"cpu",                     (str, which device to be used for the model)
  "lr": 2e-4,                                   (float, the learning rate)
  "epochs": 100,                                (int, the number of epochs)
  "batch_size" : 200,                           (int, the size of each batch of data)
  "print_freq": 100,                            (int, how often to print metrics for the trainint set)
  "eval_freq" : 30,                             (int, how often to evaluate the model and print metrics for the validation set)
  "robustness_loss": "compas_robustness_loss",  (str, the name of the robustness loss function from the losses package)
  "robust_reg": 1e-1,                           (float, the robustness regularization hyperparameter)
  "concept_reg": 1,                             (float, the concept regularization hyperparameter)
  "sparsity_reg": 2e-5,                         (float, the sparsity regularization hyperparameter)
  "manual_seed": 42                             (int, the seed to be used for reproducibility)
  "accuracy_vs_lambda": ['c1.json','c2.json']   (list of str or list of lists where the inner lists need to have the same lengths, containing the name of the config files for the accuracy vs lambda plots)
  "num_seeds": 1                                (int, number of seeds used for the accuracy_vs_lambda plot, needs to be equal to the lengths of the inner lists passed in accuracy_vs_lambda, default = 1)
}
```
Note: It is also possible to specify the architectures of the parameterizer and conceptizer classes using *config* parameters. However, to keep it neat, these are not shown here. For more information, please refer to the docstrings of the specific classes and the parameters they can take.


## Results
TBD

## Authors
* Khoi Nguyen (khoinguyen@college.harvard.edu)
* Saketh Mynampati (sbmynampati@college.harvard.edu)

**Professor:**  
Finale Doshi-Velez

## References
[1] David Alvarez Melis, Tommi S. Jaakkola  
["Towards Robust Interpretability with Self-Explaining Neural Networks"](https://papers.nips.cc/paper/2018/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html) NIPS 2018  

[2] Irina Higgins, et al.  
["β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"](https://openreview.net/forum?id=Sy2fzU9gl) ICLR 2017

[3] SENN Implementation Review with Extensions
[GitHub Repository](https://github.com/AmanDaVinci/SENN/tree/master)
