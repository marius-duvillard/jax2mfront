import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import yaml


def load_from_hdf5(
    path: Path, mode: str
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict, dict, dict]:
    """
    Charge les données et les scalers depuis un fichier HDF5.
    Retourne :
        strain, stress, stress_noisy, noise, iv, metadata, scalers, config
    """

    with h5py.File(path, "r") as f:
        group = f[mode]
        strain = group["strain"][:]
        stress = group["stress"][:]
        iv = group["internal_variables"][:] if "internal_variables" in group else None
        t = group["t"][:] if "t" in group else None

        # --- Données bruitées optionnelles ---
        stress_noisy = group["stress_noisy"][:] if "stress_noisy" in group else None
        noise = group["noise"][:] if "noise" in group else None

        # Récupérer les colonnes si elles existent
        strain_columns = (
            list(group["strain"].attrs["columns"].astype(str))
            if "columns" in group["strain"].attrs
            else None
        )
        stress_columns = (
            list(group["stress"].attrs["columns"].astype(str))
            if "columns" in group["stress"].attrs
            else None
        )
        iv_columns = (
            list(group["internal_variables"].attrs["columns"].astype(str))
            if iv is not None and "columns" in group["internal_variables"].attrs
            else None
        )

        # Métadonnées de base
        metadata = {
            "mode": group.attrs.get("mode", "unspecified"),
            "n_paths": group.attrs.get("n_paths", None),
            "n_points": group.attrs.get("n_points", None),
            "strain_columns": strain_columns,
            "stress_columns": stress_columns,
            "iv_columns": iv_columns,
        }

        # Lecture des scalers si présents
        scalers = {}
        if "scaler" in group:
            scaler_grp = group["scaler"]
            for key in scaler_grp:
                scaler = scaler_grp[key]
                if scaler.shape == ():  # scalaire
                    scalers[key] = scaler[()]  # lire la valeur directement
                else:
                    scalers[key] = scaler[:]  # lire le tableau

        # Chargement des materiaux YAML
        materials = []
        if "materials" in group:
            materials_grp = group["materials"]
            # chaque dataset est un string YAML
            for key in sorted(materials_grp.keys(), key=lambda s: int(s)):
                yaml_str = materials_grp[key][()].decode()
                materials.append(yaml.safe_load(yaml_str))

        # --- Chargement du material_index ---
        if "material_index" in group:
            material_index = group["material_index"][:]
        else:
            material_index = None

        materials_info = {
            "materials": materials,
            "material_index": material_index,
        }

        # Lecture de la config YAML si présente
        config = None
        if "config_yaml" in f:
            yaml_str = f["config_yaml"][()]
            config = yaml.safe_load(yaml_str)

    return (
        strain,
        stress,
        stress_noisy,
        noise,
        iv,
        t,
        metadata,
        scalers,
        config,
        materials_info,
    )
