#!/bin/bash

export JAX2MFRONT_FOLDER=$(dirname $(realpath $BASH_SOURCE))

# tensorflow dependencies
export LIBTENSORFLOW_ROOT=$JAX2MFRONT_FOLDER/dependencies/libtensorflow-2.6.0

export CPPFLOW_ROOT=$JAX2MFRONT_FOLDER/dependencies/cppflow
export NN_DIRECTORY=$JAX2MFRONT_FOLDER/models

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBTENSORFLOW_ROOT/lib
export LIBRARY_PATH=$LIBRARY_PATH:$LIBTENSORFLOW_ROOT/lib