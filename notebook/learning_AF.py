from jaxmat.tensors.utils import safe_norm, safe_sqrt
from jaxmat.utils import default_value, enforce_dtype
from jaxmat.tensors import dev
import pandas as pd
import jax

import jax.numpy as jnp
import equinox as eqx
from jaxmat.utils import enforce_dtype, partition_by_node_names
from jaxmat.tensors import SymmetricTensor2, main_invariants
from jaxmat.nn.icnn import ICNN
from jaxmat.state import AbstractState, make_batched, SmallStrainState
import matplotlib.pyplot as plt
from jaxmat.solvers import NewtonTrustRegion
from pt2mfront.utils.utils_hdf5 import load_from_hdf5
import jaxmat.materials as jm
import optax
from pathlib import Path

from jaxmat.materials.viscoplastic_flows import (
    ArmstrongFrederickHardening,
    NortonFlow,
    VoceHardening,
)
from jaxmat.materials.viscoplasticity import ArmstrongFrederickViscoplasticity

TRUE = dict(
    E=196_000.0,
    nu=0.3,
    sig0=150.0,
    sigu=300.0,
    b=15.0,
    C1=50_000.0,
    gamma1=300.0,
    C2=20_000.0,
    gamma2=100.0,
    K=10.0,
    m=8.0,
)


def make_material(p):
    return ArmstrongFrederickViscoplasticity(
        elasticity=jm.LinearElasticIsotropic(E=p["E"], nu=p["nu"]),
        yield_stress=VoceHardening(sig0=p["sig0"], sigu=p["sigu"], b=p["b"]),
        kinematic_hardening=ArmstrongFrederickHardening(
            C=jnp.array([p["C1"], p["C2"]]),
            gamma=jnp.array([p["gamma1"], p["gamma2"]]),
        ),
        viscous_flow=NortonFlow(K=p["K"], m=p["m"]),
    )


material_true = make_material(TRUE)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CHARGEMENT : 2 cycles ±1.5 %, 60 pas/demi-cycle
# ═══════════════════════════════════════════════════════════════════════════════
eps_max = 0.015
n_cycles = 1
Nsteps = 60

half = jnp.linspace(0, eps_max, Nsteps + 1)
cycle = jnp.concatenate([half, half[::-1][1:], -half[1:], -half[::-1][1:]])
eps_path = jnp.tile(cycle[1:], n_cycles)
dt_val = float(eps_max / (1e-3 * Nsteps))
print(f"Chargement : {len(eps_path)} pas, {n_cycles} cycles, dt = {dt_val:.3f} s")
times = jnp.arange(len(eps_path)) * dt_val
times = times.reshape(1, -1)

from jaxmat.loader import ImposedLoading, global_solve


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  INTÉGRATION DIFFÉRENTIABLE (lax.scan)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_evolution(material, eps_path, dt):
    """Intègre la loi sur le chemin eps_path (chargement mixte).
    Retourne σ_xx à chaque pas. Différentiable par rapport aux paramètres."""
    state0 = material.init_state(1)

    def step(carry, eps_xx_val):
        Eps, state = carry
        loading = ImposedLoading(
            epsxx=jnp.ones(1) * eps_xx_val,
            sigyy=jnp.zeros(1),
            sigzz=jnp.zeros(1),
        )
        Eps_new, state_new, _ = global_solve(Eps, state, loading, material, dt)
        return (Eps_new, state_new), (state_new.stress, state_new.strain)

    _, (sig, strain) = jax.lax.scan(step, (state0.strain, state0), eps_path)
    return sig, strain


compute_jit = eqx.filter_jit(compute_evolution)
print("Génération des données de référence...")
sig_clean, strain_clean = compute_jit(material_true, eps_path, dt_val)
key = jax.random.PRNGKey(42)
noise_lvl = 0.02
sig_noisy = (
    sig_clean
    + jax.random.normal(key, sig_clean.shape) * jnp.max(jnp.abs(sig_clean)) * noise_lvl
)
# print(f"  σ_xx ∈ [{float(sig_noisy.min()):.1f}, {float(sig_noisy.max()):.1f}] MPa")

strain = strain_clean.array.transpose(1, 0, 2) / jnp.mean(
    jnp.abs(strain_clean[..., 0, 0])
)
stress = sig_clean.array.transpose(1, 0, 2) / jnp.mean(jnp.abs(sig_clean[..., 0, 0]))

N_step = 500
Step = 1
train_strain = SymmetricTensor2(array=strain[:, Step:N_step:Step, ...])
train_stress = SymmetricTensor2(array=stress[:, Step:N_step:Step, ...])
times = times[:, :N_step:Step]

from optax.tree_utils import tree_scale, tree_sub, tree_add


# define evolution function
@eqx.filter_jit
def compute_evolution(material, epsilons, times, return_state=True):
    # Initial material state
    state0 = material.init_state()
    dt = jnp.diff(times)

    def step(state, args):
        new_epsilon_, dt_ = args

        # Update material response
        new_stress, new_state = material.constitutive_update(new_epsilon_, state, dt_)
        if return_state:
            return new_state, (new_stress, new_state)
        else:
            return new_state, new_stress

    # Use lax.scan to loop over gamma_list efficiently
    if return_state:
        _, (sig, state) = jax.lax.scan(step, state0, (epsilons, dt))
        return sig, state
    else:
        _, sig = jax.lax.scan(step, state0, (epsilons, dt))
        return sig


batched_compute_evolution = jax.vmap(
    compute_evolution,
    in_axes=(None, 0, 0, None),
)


class InternalState(AbstractState):
    """
    Internal state for hardening viscoplasticity
    """

    p: jax.Array = eqx.field(init=False)
    """scalar """
    epsp: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())
    """Plastic strain tensor"""
    Nvar: int = eqx.field(static=True, default=1)

    def __post_init__(self):
        # Initialise to zero
        self.p = jnp.zeros((self.Nvar,))


int_var = InternalState(Nvar=1)  # I use 1 additional scalar variable.
print(int_var)


class FreeEnergy(eqx.Module):
    elasticity: jm.AbstractLinearElastic  # Hooke isotrope
    nn_anelastic: ICNN  # ICNN for epsp variable
    nn_hardening: ICNN  # ICNN for p variable

    def __call__(self, eps, isv):
        """
        Compute free energy:
        psi = 0.5*(eps - sum_i alpha_p[i]) : C : (eps - sum_i alpha_p[i]) + NN(epsp) + NN(q)
        """
        epsp = isv.epsp
        psi_elastic = 0.5 * jnp.trace((eps - epsp) @ (self.elasticity.C @ (eps - epsp)))

        _, I2, _ = main_invariants(isv.epsp)
        psi_anelastic = (
            self.nn_anelastic(I2)
            - self.nn_anelastic(jnp.zeros_like(I2))
            - jax.jvp(self.nn_anelastic, (jnp.zeros_like(I2),), (I2,))[1]
        )
        psi_hardening = (
            self.nn_hardening(isv.p)
            - self.nn_hardening(jnp.zeros_like(isv.p))
            - jax.jvp(self.nn_hardening, (jnp.zeros_like(isv.p),), (isv.p,))[1]
        )

        # énergie totale
        return psi_elastic + psi_hardening + psi_anelastic


class DissipationPotentialQuad(eqx.Module):
    log_visco_quad: float = enforce_dtype()
    nn_corr: ICNN

    def icnn_potential(self, A_an, A_p=None):
        """
        A_an : SymmetricTensor2 (batch Nvar, 3,3)
        A_p : jnp.ndarray (batch Nvar,) ou None
        """
        # Calcul invariants principaux du batch de A
        _, I2, _ = main_invariants(A_an)  # shape (Nvar,)

        # Construit l'entrée du réseau
        if A_p is not None:
            I2 = jnp.atleast_1d(I2)
            A_p = jnp.atleast_1d(A_p)
            x = jnp.concatenate([I2, A_p], axis=0)
        else:
            x = I2
        return self.nn_corr(x)

    @property
    def visco_quad(self):
        return jnp.exp(self.log_visco_quad)

    def __call__(self, isv):
        """
        Compute free energy:
        phi = ICNN(I2(\dot epsp), \dot p)
        """
        A_an = isv.epsp.tensor
        A_p = isv.p
        I1, I2, I3 = main_invariants(A_an)
        # s_A_an = jnp.sqrt(3 / 2.0) * safe_norm(dev(A_an))
        phi_quad = 0.5 * I2**2 / self.visco_quad
        phi = self.icnn_potential(A_an, A_p)
        return (
            phi_quad + phi
        )  # - jax.jvp(lambda a: self.icnn_potential(SymmetricTensor2(tensor=a), A_p=jnp.zeros_like(A_p)),(jnp.zeros_like(A_p),),(A_p,),)[1]


class ICNNDissipationPotential(ICNN):
    def icnn_potential(self, A, Q=None):
        """
        A : SymmetricTensor2 (batch Nvar, 3,3)
        Q : jnp.ndarray (batch Nvar,) ou None
        """
        # Calcul invariants principaux du batch de A
        # I1, I2, I3 = jax.vmap(main_invariants)(A)  # shape (Nvar,)
        I1, I2, I3 = main_invariants(A)
        # Construit l'entrée du réseau
        if Q is not None:
            I2 = jnp.atleast_1d(I2)
            Q = jnp.atleast_1d(Q)
            x = jnp.concatenate([I2, Q])
        else:
            x = I2
        return super().__call__(x)

    def __call__(self, isv):
        """
        isv: objet avec
            .A : SymmetricTensor2 (forces sur alpha_p)
            .Q : jnp.ndarray (forces sur q)
        """
        A_p = isv.epsp.tensor
        Q = isv.p

        # Potentiel pseudo-dissipatif dual
        return (
            self.icnn_potential(A_p, Q)
            - self.icnn_potential(jnp.zeros_like(A_p), jnp.zeros_like(Q))
            - jax.jvp(
                lambda a: self.icnn_potential(
                    SymmetricTensor2(tensor=a), Q=jnp.zeros_like(Q)
                ),
                (jnp.zeros_like(A_p),),
                (A_p,),
            )[1]
            - jax.jvp(
                lambda q: self.icnn_potential(
                    SymmetricTensor2(tensor=jnp.zeros_like(A_p)), Q=q
                ),
                (jnp.zeros_like(Q),),
                (Q,),
            )[1]
        )


hidden_dims = [8, 8]  # (16,16,16)
Nvar = 1
key = jax.random.PRNGKey(42)
key1, key2, key3 = jax.random.split(key, num=3)  # define keys for reproducibility
E_scale = (
    196000
    * jnp.mean(jnp.abs(strain_clean[..., 0, 0]))
    / jnp.mean(jnp.abs(sig_clean[..., 0, 0]))
)  # 1
nu = 0.3
elasticity = jm.LinearElasticIsotropic(E=E_scale, nu=nu)

nn_anelastic = ICNN(1, hidden_dims, key1)  # one input the second main invariant
nn_hardening = ICNN(Nvar, hidden_dims, key2)  # Nvar inputs
free_energy = FreeEnergy(elasticity, nn_anelastic, nn_hardening)
# dissipation_potential = DissipationPotential(Nvar + 1, hidden_dims, key3)
icnn_corr = ICNN(Nvar + 1, hidden_dims, key3)
visco_quad = 3
# dissipation_potential = DissipationPotentialQuad(visco_quad, icnn_corr)
# dissipation_potential = DissipationPotential(Nvar + 1, hidden_dims, key3)
dissipation_potential = ICNNDissipationPotential(Nvar + 1, hidden_dims, key3)


class GSM(jm.GeneralizedStandardMaterialDual):
    def make_internal_state(self):
        return InternalState(Nvar=Nvar)


gsm = GSM(free_energy, dissipation_potential)
state0 = gsm.init_state()

state = gsm.init_state()
# using all the data
times_train = times[:, :]
sig_train, state_train = batched_compute_evolution(gsm, train_strain, times_train, True)
times.shape

sig_noise = SymmetricTensor2(tensor=train_stress)  # first train on one path

learning_rate = 5e-2
max_steps = 100
optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_param_block_rms(),
    optax.scale(-learning_rate),
)


# --- Loss function ---
@eqx.filter_jit
def loss_fn(params, args):
    sig_data, static_ = args
    material_ = params if static_ is None else eqx.combine(params, static_)
    sig_hat = batched_compute_evolution(material_, train_strain, times_train, False)
    diff = sig_hat.array - sig_data.array
    loss_val = 0.5 * jnp.mean(diff**2)
    return loss_val, sig_hat


# --- Gradient step ---
@jax.jit
def step(params, opt_state, args):
    (loss_val, sig_hat), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, args)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val, sig_hat


# --- Boucle sur gsm / Nvar ---
trainable, static = partition_by_node_names(gsm, ["free_energy.elasticity"])

# initialisation optimizer
params = trainable
opt_state = optimizer.init(params)

# Training loop
for step_idx in range(max_steps):
    params, opt_state, loss_val, sig_hat_train = step(
        params, opt_state, (sig_noise, static)
    )

    if step_idx % 1 == 0:
        print(f"Step {step_idx}: loss = {float(loss_val):.6e}")

# Combiner trainable + static pour obtenir le matériau complet
trained_material = params if static is None else eqx.combine(params, static)

for i in range(5):
    label_gt = "gt" if i == 0 else None
    label_train = "train" if i == 0 else None
    plt.scatter(times_train[i, 1:].T, train_stress[i, :, 0, 1].T, label=label_gt)
    plt.plot(times_train[i, 1:].T, sig_hat_train[i, :, 0, 1].T, label=label_train)
plt.xlabel("t")
plt.ylabel("$\sigma_{xy}$")

plt.legend()
plt.show()
