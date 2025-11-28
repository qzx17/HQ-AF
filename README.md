# hqaf

We will introduce how to use the code and datasets in the folder to train and test our Hierarchical Question Attribute-Fused Knowledge Tracing (HQAF-KT) model. Our experiment procedures follow the paradigm provided by a knowledge tracing library, [PyKT](https://github.com/pykt-team/pykt-toolkit). 

## Directory Overview

This directory contains the following three subdirectories:

- `configs` → Configuration Files
  - This directory contains configuration files. These files define the settings and parameters for the program to ensure it runs according to the specified requirements.

- `examples` → Executable Files
  - This directory contains executable files and example code.

- `data` → Database Files
  - This directory contains database files. The data files are used to store the data required for the program to run, including input data and output results.


## Packages Installation
To set up the environment for training and testing the hqaf model, you need to install the following Python packages. You can install them using `pip` with the specified versions to ensure compatibility: 

```
pip install torch==2.4.0 pandas==2.2.2 scikit-learn numpy==1.26.4
```

## Dataset Preparation

We use three datasets in our paper: ASSIST2009，ASSIST2017 and AAAI2023.  First, we should change directory to `examples`:

```bash
cd examples
```

Then we could prepare the dataset, we take ASSIST2009 for example:

```python
python data_preprocess.py --dataset_name=assist2009
```

We can use `--dataset_name=assist2017` for the dataset ASSIST2017 and `--dataset_name=aaai2023` for the dataset AAAI2023.

## Train Model

After the data preprocessing, you can use the `python hqaf_train.py [parameter]` to train the model, we also take `assist2009` for example:

```python
python hqaf_train.py --dataset_name=assist2009 --num_attn_heads=4
```

`--num_attn_heads` is used to set the number of attention heads and the default value is four.

## Test Model

The model is saved by default in the `saved_model` directory, and the model name will be printed to the console. Set the parameter `--save_dir` to the model name. 

```python
python hqaf_test.py --save_dir=saved_model/aaai2023/aaai2023_hqaf_qid_saved_model_3407_0_0.2_256_512_4_4_0.0004_0_0_for_test

