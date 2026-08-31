# CLINE.md - JAX2MFront Project Guide

## Overview

This project provides tools to train neural network constitutive laws using JAX/JAXMAT and convert them to TensorFlow for use with the MFront framework (TFEL).

## Project Structure

```
jax2mfront/
├── data/                    # Training datasets (HDF5 format)
├── dependencies/
│   └── jaxmat/             # Forked JAXMAT library with modifications
├── mfront/                  # MFront behaviour definitions
│   ├── gsm_norton/         # Norton viscoplasticity
│   ├── gsm_norton_hardening/  # Norton with hardening
│   ├── gsm_norton_iso_kin/    # Norton with isotropic/kinematic hardening
│   ├── gsm_visco/          # GSM visco-plasticity
│   ├── gsm_visco_branch/   # GSM with branching
│   ├── nl_elasticity/      # Non-linear elasticity
│   └── test.mfront         # Test behaviour
├── model/                   # Converted TensorFlow models
│   ├── norton_nn/          # Norton model
│   ├── norton_hardening_nn/ # Norton with hardening
│   ├── gsm_model/          # GSM model
│   └── ...
├── train_convert_*.ipynb   # Jupyter notebooks for training/conversion
├── gsm_jaxmat_*.ipynb      # Jupyter notebooks for specific GSM models
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
└── env.sh                  # Environment setup script
```

## Installation

### Python Environment

```bash
python3.10 -m venv .venv/jax2mfront
source .venv/jax2mfront/bin/activate
pip install -r requirements.txt
pip install -e dependencies/jaxmat
```

### Cpp/MFront Dependencies

```bash
# Source the environment
. env.sh

# Install libtensorflow (C++ API)
cd dependencies
mkdir libtensorflow-2.6.0
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-linux-x86_64-2.6.0.tar.gz
tar -C . -xzf libtensorflow-linux-x86_64-2.6.0.tar.gz

# Clone cppflow
cd dependencies
git clone https://github.com/serizba/cppflow
```

## Training Neural Network Models

### 1. Non-Linear Elasticity

```bash
# Train and convert model
jupyter nbconvert --to notebook --execute train_convert_nlelastic.ipynb
```

Model: Non-linear elastic law using a neural network to learn the stress-strain relationship.

### 2. GSM Visco-Plasticity (GSM)

```bash
# Train GSM visco-plasticity model
jupyter nbconvert --to notebook --execute train_convert_visco.ipynb
```

Model: GSM formulation with two potentials (free energy Ψ and dissipation potential Φ*) using ICNN.

### 3. GSM with Branching

```bash
# Train GSM with branching model
jupyter nbconvert --to notebook --execute train_convert_visco2.ipynb
```

Model: Extended GSM formulation with branching internal variables.

### 4. Norton Visco-Plasticity

```bash
# Train Norton visco-plasticity model
jupyter nbconvert --to notebook --execute gsm_jaxmat_norton.ipynb
```

Model: Norton-type visco-plasticity following the GSM formulation.

### 5. Norton with Kinematic Hardening

```bash
# Train Norton with kinematic hardening model
jupyter nbconvert --to notebook --execute gsm_jaxmat_norton_hardening.ipynb
```

Model: Norton visco-plasticity with kinematic hardening using neural networks.

## Converting JAX Models to TensorFlow

The training notebooks perform the following conversion steps:

1. Train the model using JAX/Equinox
2. Export to TensorFlow SavedModel format using `jax2tf`
3. Save the model with proper signatures for MFront integration

Example conversion code pattern:

```python
import jax2tf
import tensorflow as tf

# After training with Equinox
model = ...  # Trained Equinox model

# Convert to TensorFlow
converter = jax2tf.convert(
    lambda x: model.apply(x),
    enable_xla=False,
    native_serialization=True
)

# Create concrete function
concrete_func = tf.function(
    converter,
    autograph=False
).get_concrete_function(
    tf.TensorSpec(input_shape, tf.float64)
)

# Save model
tf.saved_model.save(
    model, 
    output_dir,
    signatures=concrete_func
)
```

## MFront Compilation

### Norton Visco-Plasticity

```bash
cd mfront/gsm_norton
mfront \
  -I $CPPFLOW_ROOT/include \
  -I $LIBTENSORFLOW_ROOT/include \
  --obuild --interface=generic \
  norton_nn.mfront
```

### Norton with Hardening

```bash
cd mfront/gsm_norton_hardening
mfront \
  -I $CPPFLOW_ROOT/include \
  -I $LIBTENSORFLOW_ROOT/include \
  --obuild --interface=generic \
  norton_hardening_nn.mfront
```

### Norton with Isotropic/Kinematic Hardening

```bash
cd mfront/gsm_norton_iso_kin
mfront \
  -I $CPPFLOW_ROOT/include \
  -I $LIBTENSORFLOW_ROOT/include \
  --obuild --interface=generic \
  norton_iso_kin_nn.mfront
```

### GSM Visco-Plasticity

```bash
cd mfront/gsm_visco
mfront \
  -I $CPPFLOW_ROOT/include \
  -I $LIBTENSORFLOW_ROOT/include \
  --obuild --interface=generic \
  ngsm.mfront
```

## MTest Evaluation

```bash
# Run relaxation test
mtest gsm_norton/relax.mtest
mtest gsm_norton_hardening/relax.mtest
mtest gsm_visco/relax.mtest

# Plot results
python gsm_norton/plot_relax.py
python gsm_visco/plot_relax.py
```

## Model Types

### 1. Non-Linear Elasticity
- **Input**: Strain tensor (6 components)
- **Output**: Stress tensor (6 components)
- **Network**: Feed-forward neural network
- **Use Case**: Hyperelastic materials with non-linear stress-strain response

### 2. GSM Visco-Plasticity
- **Input**: Strain tensor, internal variables, time step
- **Output**: Stress tensor, internal variable rates
- **Network**: ICNN for dissipation potential Φ*
- **Use Case**: Time-dependent plasticity with viscous effects

### 3. Norton Visco-Plasticity
- **Input**: Strain tensor, viscoplastic strain, time step
- **Output**: Stress tensor, viscoplastic strain rate
- **Network**: Neural network for viscosity surface
- **Use Case**: Power-law visco-plasticity (Norton flow)

### 4. Norton with Hardening
- **Input**: Strain tensor, internal variables (hardening), time step
- **Output**: Stress tensor, internal variable rates
- **Network**: Neural network for hardening evolution
- **Use Case**: Visco-plasticity with isotropic/kinematic hardening

## File Format

### Training Data (HDF5)
- `strain`: Strain tensor history (n_paths, n_steps, 6)
- `stress`: Stress tensor history (n_paths, n_steps, 6)
- `times`: Time history (n_paths, n_steps)
- `scalers`: Normalization parameters
- `config`: Model configuration
- `material_infos`: Material parameters

### TensorFlow Model (SavedModel)
- `saved_model.pb`: Model definition
- `variables/`: Model weights
- `assets/`: Additional files

## MFront Integration

The MFront laws use:
1. `cppflow` for TensorFlow C++ API
2. Custom tensor conversions (`to_stensor`, `to_st2tost2`)
3. Neural network inference in MFront code
4. Automatic differentiation support

Model path configuration in MFront:
```cpp
inline NN &getNN()
{
    static NN nn("../../model/model_name");
    return nn;
}
```

## Environment Variables

```bash
# Source env.sh for TFEL environment
. env.sh

# Required for MFront compilation
export TFEL_HOME=/path/to/tfel
export CPPFLOW_ROOT=/path/to/cppflow
export LIBTENSORFLOW_ROOT=/path/to/libtensorflow
```

## Troubleshooting

### Common Issues

1. **TensorFlow version mismatch**: Ensure libtensorflow version matches the Python tensorflow version
2. **Missing symbols during MFront compilation**: Verify library paths in `@Link` directive
3. **Model loading errors**: Check model path in MFront code matches actual location

### Verification

```bash
# Check TensorFlow installation
python -c "import tensorflow; print(tensorflow.__version__)"

# Check MFront compilation
mfront --version

# Verify model signature
saved_model_cli show --dir model/norton_nn/ --tag_set serve --signature_def serving_default
```

## References

- JAXMAT: https://github.com/ihm-departement/jaxmat
- MFront: https://tfel.sourceforge.net/mfront.html
- TFEL: https://tfel.sourceforge.net/