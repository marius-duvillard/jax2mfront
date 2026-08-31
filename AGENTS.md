# AGENTS.md — jax2mfront

## What this repo does

Train neural-network constitutive laws with **JAX/Equinox**, convert them to **TensorFlow SavedModel**, and integrate into **MFront (TFEL)** via `cppflow`. One Python file (`generate_mfront.py`) assembles `.mfront` behaviour files from JSON configs.

---

## Top-level structure

| Path | Purpose |
|------|---------|
| `notebook/*.ipynb` | Training + conversion notebooks (**source of truth** for how models are trained/converted) |
| `materials.py` | `GeneralizedStandardMaterialDual`, `GeneralizedStandardMaterialDualParam` — GSM base classes built on `jaxmat.materials` |
| `mfront_nn/generate_mfront.py` | JSON→MFront code generator (the only tool for assembling `.mfront`) |
| `mfront_nn/mfront_nn_toolkit/` | Forked copy of the toolkit — configs, generated examples, generic header |
| `mfront/*/` | Compiled MFront behaviour directories (each sub-folder = one constitutive law: `gsm_norton/`, `gsm_visco/`, etc.) |
| `model/*/` | Pre-trained TensorFlow SavedModel files per constitutive law |
| `dependencies/jaxmat/` | **Forked** jaxmat library (required to install) |
| `env.sh` | Sets `CPPFLOW_ROOT`, `LIBTENSORFLOW_ROOT`, `LD_LIBRARY_PATH`, TFEL paths — **must source before any MFront work** |

---

## Environment setup

```bash
# Python (3.10)
python3.10 -m venv .venv/jax2mfront && source .venv/jax2mfront/bin/activate
pip install -r requirements.txt
pip install -e dependencies/jaxmat          # forked jaxmat required

# C++ / MFront
source env.sh                               # sets CPPFLOW_ROOT, LIBTENSORFLOW_ROOT, etc.
# Then ensure libtensorflow and cppflow exist in:
#   dependencies/libtensorflow-2.6.0/
#   dependencies/cppflow/
```

**Never skip `source env.sh` before MFront compilation — `$CPPFLOW_ROOT` and `$LIBTENSORFLOW_ROOT` will be missing.**

---

## Running notebooks (train → TF export)

```bash
jupyter nbconvert --to notebook --execute noteboook/<notebook-name>.ipynb
```

Notebooks run with `jax_platform_name = "cpu"` by default. Enable GPU by changing the config line.

**Training data**: HDF5 files in `data/` (loaded via `load_from_hdf5` in `notebook/utils.py`).

---

## Adding a new constitutive law

1. **Train** in a JAX notebook → save as TensorFlow SavedModel to `model/<name>/`.
2. **Inspect signatures**: `saved_model_cli show --dir model/<name>/ --tag_set serve --all` — verify input/output tensor names (they become `tf_input` / `tf_output` in the config).
3. **Create config**: `mfront_nn/mfront_nn_toolkit/configs/<law>.json` — one JSON per behavior. Pick an existing template from `INTEGRATOR_GENERATORS` in `generate_mfront.py` as the starting point for `integrator_template`.
4. **Generate .mfront**:
   ```bash
   python3 mfront_nn/generate_mfront.py mfront_nn/mfront_nn_toolkit/configs/<law>.json -o mfront/<dir>/<law>.mfront
   # or preview:  --dry-run
   ```
5. **Compile MFront** (from the target directory):
   ```bash
   source ~/path/to/env.sh  # or wherever env.sh lives
   mfront -I $CPPFLOW_ROOT/include \
          -I $LIBTENSORFLOW_ROOT/include \
          --obuild --interface=generic \
          <law>.mfront
   ```
6. **Evaluate** with MTest (if available): `mtest <dir>/<test-file>.mtest`

---

## Adding a new integrator template

In `mfront_nn/generate_mfront.py`, find `INTEGRATOR_GENERATORS`. Add:

```python
def gen_integrator_<name>(cfg: dict) -> str:
    return """@Integrator { ... }""";  # uses {{model_path}}-like substitutions from cfg
```

Then register it in the `INTEGRATOR_GENERATORS` dict. The generator receives the full config dict; use `cfg["tensorflow"]["lib_path"]`, `cfg["behaviour"]`, etc. as needed.

---

## Key gotchas

- **jaxmat is forked**: Do not `pip install jaxmat` from PyPI — it will miss corrections. Always use `pip install -e dependencies/jaxmat`.
- **tensorflow C++ API version must match Python tensorflow version**: TF 2.20.0 in `requirements.txt`; C++ lib in `dependencies/libtensorflow-2.6.0/` is an example — update both if changing versions.
- **MFront code paths**: The `@Includes` block in generated `.mfront` files references `model_path` relative to the MFront source location. If you move directories, fix paths in the JSON config's `tensorflow.model_path`.
- **Generated `.mfront` files contain embedded C++** (the `NN` struct + `getNN()` singleton). They are not minimal — they include full tensor conversions via cppflow.
- **Use `nn_mfront.hxx` or an external header** (`use_external_header` in config) when you want to share a single NN interface across multiple `.mfront` files instead of embedding C++ inline.

---

## Test / verification commands

```bash
# Verify TensorFlow model signatures (after training/export):
saved_model_cli show --dir model/<name>/ --tag_set serve --signature_def serving_default

# Verify MFront compiles:
cd mfront/<dir> && mfront -I $CPPFLOW_ROOT/include -I $LIBTENSORFLOW_ROOT/include --obuild --interface=generic <law>.mfront

# Run relaxation tests (if MTest available):
mtest gsm_norton/relax.mtest
python gsm_norton/plot_relax.py    # plots comparison with ground truth
```

---

## References / external docs

- **jaxmat**: `dependencies/jaxmat/` (forked repo — read its README there)
- **MFront DSL**: Refer to TFEL docs; key directives used here: `@DSL`, `@Behaviour`, `@Integrator`, `@ComputeFinalStress`, `@TangentOperator`, `@StateVariable`, `@Link`, `@Includes`
- **cppflow**: `dependencies/cppflow/` — C++ wrapper for TensorFlow runtime
- **CLINE.md** in root: more detailed project guide (older than this file; use as supplement)
