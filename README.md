# Neural network constitutive law using Jaxmat and MFront

This repository explain how to train a neural network in order to predict material constitutive behaviour inside the MFront Framework.

# Installation

## Python
A python environment need to be installed with jax and jaxmat to train the neural network. We recommand to use the following commands.

> If you are only interesed by using the MFront part, a non linear elastic model have added in the model folder. You can directly go to the section [Cpp/MFront](#cppmfront-dependencies).
First create a venv with the required package (jax and tensorflow)
```sh
python3.10 -m venv <venv_folder>
source venv/bin/activate
pip install -r requirements.txt
```

Due to minor correction, we fork the jaxmat library and add it inside the dependencies folder. The installation is then simply
```sh
pip install -e dependencies/jaxmat
```

## Cpp/MFront dependencies 

In order to use the Neural Network model with MFront, cpp APIs (cppflow/libtensorflow) are used.
First source the `env.sh` file

```sh
. env.sh
```

We first install libtensorflow (the cpp API of tensorflow). Download it on [this page](https://www.tensorflow.org/install/lang_c?hl=fr) with the right OS and extract in the dependencies folder

```sh
cd dependencies
mkdir libtensorflow-2.6.0
wget <file_link>
tar -C . -xzf <downloaded_file>
```

We then install the cppflow headers. For instance, you can simply clone [this repository](https://github.com/serizba/cppflow) inside the dependencies folder. 

Moreover, the TFEL environment must be installed. Follow up this installation procedure.


# Training

Based on data, we define and train our model thanks to the jaxmat library. This library uses jax a python library that natively use autodifferentiation, which is propice for deep learning as well as scientific computing.

First you need to create a python environment with the needed [requirements](requirements.txt):

```bash
venv -m create .venv/jax2mfront <requirements.txt ?>
. .venv/jax2mfront/bin/activate

```

The notebook [convert_model.ipynb](convert_model.ipynb) explains how to train a model to learn a non-linear elastic law. Then how to convert the model equinox/jax pytree model class in order to finally obtain the [nl_model](nl_model) repository with all the needed neural network graphs and weights. 

# MFront law

We defined the MFront law in the [mfront folder](mfront/nl-elasticity.mfront).

```bash
cd mfront
mfront \
  -I $CPPFLOW_ROOT/include \
  -I $LIBTENSORFLOW_ROOT/include \
  --obuild --interface=generic\
  nl-elasticity.mfront
```

You will also find the MFront law that have been used to generate the training data.

# MTest evaluation

We implement a simple MTest for material point test. 

