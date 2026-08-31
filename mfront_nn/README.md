# mfront_nn_toolkit

Outil de génération automatique de fichiers MFront pour l'intégration de réseaux de neurones TensorFlow (entraînés sous JAX/Equinox, convertis en SavedModel TF).

## Structure

```
mfront_nn_toolkit/
├── generate_mfront.py       ← Générateur principal
├── include/
│   └── nn_mfront.hxx        ← Header C++ générique (conversions TFEL ↔ cppflow)
├── configs/                 ← Configurations JSON, une par comportement
│   ├── norton_nn.json
│   ├── norton_hardening_nn.json
│   ├── norton_iso_kin_nn.json
│   ├── nl_elasticity.json
│   ├── ngsm_2.json
│   └── ngsm_branch.json
└── examples/                ← Fichiers .mfront générés (résultats)
    ├── norton_nn.mfront
    ├── ...
```

---

## Utilisation rapide

```bash
# Générer un fichier .mfront depuis une config JSON
python3 generate_mfront.py configs/norton_nn.json

# Spécifier le nom de sortie
python3 generate_mfront.py configs/ngsm_branch.json -o MonComportement.mfront

# Prévisualiser sans écrire
python3 generate_mfront.py configs/ngsm_2.json --dry-run
```

---

## Schéma de configuration JSON

### Champs globaux

| Champ                       | Type    | Description                                              |
|-----------------------------|---------|----------------------------------------------------------|
| `behaviour`                 | string  | Nom du comportement MFront                               |
| `author`                    | string  |                                                          |
| `date`                      | string  |                                                          |
| `description`               | string  |                                                          |
| `dsl`                       | string  | `"Implicit"`, `"ImplicitII"`, `"DefaultDSL"`             |
| `modelling_hypothesis`      | string  | `"Tridimensional"` (seul supporté)                       |
| `algorithm`                 | string  | `"NewtonRaphson"` (optionnel)                            |
| `theta`                     | number  | Schéma d'intégration (0.5 = Crank-Nicolson, 1 = Euler)  |
| `epsilon`                   | number  | Critère de convergence                                   |
| `profiling`                 | bool    | Active `@Profiling true`                                 |
| `provides_tangent_operator` | bool    | Ajoute `@ProvidesTangentOperator`                        |
| `integrator_template`       | string  | Nom du template d'intégrateur (voir ci-dessous)          |

### `tensorflow`

```json
"tensorflow": {
  "lib_path": "../../dependencies/libtensorflow-2.6.0/lib",
  "model_path": "../../model/mon_modele",
  "extra_link": "",              // optionnel, ex: "-L/path -lmylib"
  "use_external_header": "nn_interface.hxx"  // optionnel, remplace le bloc @Includes généré
}
```

### `standard_elasticity`

```json
"standard_elasticity": {
  "enabled": true,
  "young_modulus": 200e9,
  "poisson_ratio": 0.3
}
```

### `state_variables`

```json
"state_variables": [
  { "name": "evp",   "type": "StrainStensor", "glossary_name": "ViscoplasticStrain" },
  { "name": "p",     "type": "strain",        "entry_name": "p" },
  { "name": "alpha", "type": "StrainStensor", "array_size": 3 }
]
```

### `nn_functions`

Chaque fonction NN est définie par ses entrées et sorties TensorFlow.

```json
"nn_functions": [
  {
    "name": "phi2",
    "signature_name": "phi2",
    "inputs": [
      { "name": "sigma", "tf_input": "phi2_sigma:0", "type": "stensor" },
      { "name": "p",     "tf_input": "phi2_p:0",     "type": "scalar"  }
    ],
    "outputs": [
      { "name": "depsp",       "tf_output": "PartitionedCall:0", "type": "stensor",  "index": 0 },
      { "name": "ddepsp_dsig", "tf_output": "PartitionedCall:1", "type": "st2tost2", "index": 1 }
    ]
  }
]
```

**Types d'entrées supportés :**

| `type`          | C++ généré                               | Forme TF   |
|-----------------|------------------------------------------|------------|
| `stensor`       | `const nmf::Stensor&`                    | `{6}`      |
| `scalar`        | `const double&`                          | `{1}`      |
| `stensor_array` | `const fsarray<N, nmf::Stensor>&`        | `{N, 6}`   |

**Types de sorties supportés :**

| `type`            | C++ généré                                  | Forme TF          |
|-------------------|---------------------------------------------|-------------------|
| `stensor`         | `nmf::Stensor`                              | `{6}`             |
| `st2tost2`        | `nmf::Stensor4`                             | `{36}`            |
| `scalar`          | `double`                                    | `{1}`             |
| `stensor_array`   | `fsarray<N, nmf::Stensor>`                  | `{N*6}`           |
| `st2tost2_array`  | `fsarray<N, nmf::Stensor4>`                 | `{N*36}`          |
| `st2tost2_matrix` | `fsarray<N*M, nmf::Stensor4>`               | `{N, M, 6, 6}`    |

---

## Templates d'intégrateur disponibles

| `integrator_template`                      | Usage                                                        |
|--------------------------------------------|--------------------------------------------------------------|
| `nl_elasticity`                            | DefaultDSL, stress+tangent depuis strain                     |
| `standard_elasticity_phi2_sigma`           | Implicit + SE, phi2(sigma) → depsp                           |
| `standard_elasticity_phi2_sigma_p`         | Implicit + SE, phi2(sigma, p) → depsp + dp                   |
| `standard_elasticity_phi2_sigma_p_alpha`   | Implicit + SE, phi2(sigma, p, alpha) → depsp + dp + dalpha   |
| `ngsm_single_alpha`                        | ImplicitII, phi1+phi2(eps, alpha) avec 1 alpha stensor       |
| `ngsm_branch_alpha`                        | ImplicitII, phi1+phi2(eps, alpha[N]) avec N branches         |

Pour ajouter un nouveau template : implémenter une fonction `gen_integrator_<nom>(cfg) -> str` dans `generate_mfront.py` et l'enregistrer dans `INTEGRATOR_GENERATORS`.

---

## Header générique `nn_mfront.hxx`

Fournit le namespace `nmf::` avec :

- `nmf::Stensor` / `nmf::Stensor4` — alias de types TFEL 3D
- `nmf::stensor_to_tf(s)` — conversion stensor → cppflow::tensor
- `nmf::stensor_array_to_tf<N>(arr)` — fsarray<N> → tensor {N,6}
- `nmf::scalar_to_tf(v)` — scalaire → tensor {1}
- `nmf::tf_to_stensor(t, r)` / `tf_to_stensor(t)` — tensor → stensor
- `nmf::tf_to_st2tost2(t, r)` / `tf_to_st2tost2(t)` — tensor → st2tost2
- `nmf::tf_to_stensor_array<N>(t, r)` — tensor {N*6} → fsarray<N,Stensor>
- `nmf::tf_to_stensor4_array<N>(t, r)` — tensor {N*36} → fsarray<N,Stensor4>
- `nmf::tf_to_stensor4_matrix<N,M>(t, r)` — tensor {N,M,6,6} → fsarray<N*M,Stensor4>
- `nmf::get_model(path)` — singleton cppflow::model
- `nmf::numerical_jacobian(f, x, h)` — différences finies centrées

---

## Workflow typique

```
1. Entraîner le réseau sous JAX/Equinox
2. Convertir en TF SavedModel (avec signatures nommées)
3. Vérifier les noms de signatures et tenseurs :
       saved_model_cli show --dir mon_model/ --tag_set serve --all
4. Créer configs/mon_comportement.json
5. python3 generate_mfront.py configs/mon_comportement.json
6. Compiler avec mfront
```

---

## Cas d'usage couverts

| Fichier original            | Config JSON               | Template                                    |
|-----------------------------|---------------------------|---------------------------------------------|
| `norton_nn.mfront`          | `norton_nn.json`          | `standard_elasticity_phi2_sigma`            |
| `norton_hardening_nn.mfront`| `norton_hardening_nn.json`| `standard_elasticity_phi2_sigma_p`          |
| `norton_iso_kin_nn.mfront`  | `norton_iso_kin_nn.json`  | `standard_elasticity_phi2_sigma_p_alpha`    |
| `nl-elasticity.mfront`      | `nl_elasticity.json`      | `nl_elasticity`                             |
| `ngsm_2.mfront`             | `ngsm_2.json`             | `ngsm_single_alpha`                         |
| `ngsm_branch.mfront`        | `ngsm_branch.json`        | `ngsm_branch_alpha`                         |
| `ngsm_w_elast.mfront`       | (use `use_external_header`)| header `nn_interface.hxx` existant         |
