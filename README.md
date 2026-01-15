# Neural Network Constitutive Law using Jaxmat and MFront

This repository explains how to train a neural network to predict material constitutive behavior within the MFront framework.

# Installation

## Python

A Python environment is required with `jax` and `jaxmat` installed to train the neural network. We recommend using the following commands.

> If you are only interested in using the MFront part, a pre-trained non-linear elastic model is already provided in the `model` folder. You can directly go to the section [Cpp/MFront](#cppmfront-dependencies).

First, create a virtual environment and install the required packages (`jax` and `tensorflow`):

```sh
python3.10 -m venv <venv_folder>
source <venv_folder>/bin/activate
pip install -r requirements.txt
```

Due to minor corrections, we forked the `jaxmat` library and included it inside the `dependencies` folder. Installation is then simply:

```sh
pip install -e dependencies/jaxmat
```

## Cpp/MFront Dependencies

To use the Neural Network model with MFront, the C++ APIs (`cppflow` / `libtensorflow`) are required. First, source the `env.sh` file:

```sh
. env.sh
```

### Installing libtensorflow (C++ API)

Download the appropriate version for your OS from [this page](https://www.tensorflow.org/install/lang_c?hl=fr) and extract it into the `dependencies` folder:

```sh
cd dependencies
mkdir libtensorflow-2.6.0
wget <file_link>
tar -C . -xzf <downloaded_file>
```

### Installing cppflow

You can clone [cppflow](https://github.com/serizba/cppflow) directly into the `dependencies` folder.

### TFEL Environment

The TFEL environment must also be installed. Please follow the official TFEL installation procedure.

# Training

Using the generated data, we define and train our model with the `jaxmat` library. This library is based on `jax`, a Python library that provides **automatic differentiation**, which is ideal for deep learning as well as scientific computing.

First, create a Python environment with the required [dependencies](requirements.txt):

```bash
python3 -m venv .venv/jax2mfront
source .venv/jax2mfront/bin/activate
pip install -r requirements.txt
```

The notebook [convert_model.ipynb](convert_model.ipynb) explains how to train a model to learn a non-linear elastic law, and how to convert the Equinox/Jax PyTree model class to produce the [nl_model](nl_model) repository containing all the required neural network graphs and weights.

# MFront Law

The MFront law is defined in the [mfront folder](mfront/nl-elasticity.mfront):

```bash
cd mfront
mfront \
  -I $CPPFLOW_ROOT/include \
  -I $LIBTENSORFLOW_ROOT/include \
  --obuild --interface=generic \
  nl-elasticity.mfront
```

You will also find the MFront law that was used to generate the training data.

# MTest Evaluation

We implemented a simple MTest for material point evaluation.
