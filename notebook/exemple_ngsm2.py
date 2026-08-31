from jaxmat.state import AbstractState
from jaxmat.tensors import SymmetricTensor2, main_invariants
import jax.numpy as jnp
from jaxmat.nn.icnn import ICNN
import jaxmat.materials as jm
import jax
import equinox as eqx

### LOAD DATA ###
"""
here you create your data sequences of shape (Nseq, Nstep, 6) or (Nstep, 6). Be sure to normalize them such as eps, sig \in [0,1]. In order to preserve isotropy, normalize with the same scale factor each columns.
"""

### DEFINE INTERNAL STATE ###
"""
We define a GSM2 model, that is to say, we separate the internal variables between a anelastic strain and other thermodynamics variables."""


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

### DEFINE THE POTENTIAL

"""
The model is define by two potentials the free_energy potential and the dissipation'potential. We separate each terms concerning the elastic part, the anelastic part and additionnal variables.
"""


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
        psi_anelastic = self.nn_anelastic(I2)
        psi_hardening = self.nn_hardening(isv.p)

        # énergie totale
        return psi_elastic + psi_anelastic + psi_hardening


class DissipationPotential(ICNN):
    def icnn_potential(self, A_an, A_p=None):
        """
        A_an : SymmetricTensor2 (batch Nvar, 3,3)
        A_p : jnp.ndarray (batch Nvar,) ou None
        """
        # Calcul invariants principaux du batch de A
        _, I2, _ = main_invariants(A_an)  # shape (Nvar,)

        # Construit l'entrée du réseau
        if A_p is not None:
            x = jnp.concatenate([I2, A_p], axis=0)
        else:
            x = I2
        return super().__call__(x)

    def __call__(self, isv):
        """
        Compute free energy:
        phi = ICNN(I2(\dot epsp), \dot p)
        """
        A_an = isv.epsp
        A_p = isv.p

        phi = self.icnn_potential(A_an, A_p)

        # and normalisation terms such as phi(0) = 0 and \partial pĥi(0) = 0
        phi_0 = self.icnn_potential(SymmetricTensor2(), jnp.zeros_like(A_p))
        # dphi_dA_p_0 = jax.jvp(
        #     lambda a: self.icnn_potential(SymmetricTensor2(a), jnp.zeros_like(A_p)),
        #     (jnp.zeros_like(A_an),),
        #     (A_p,),
        # )[1]
        # dphi_dA_an_0 = jax.jvp(
        #     self.icnn_potential(SymmetricTensor2(), lambda a: jnp.zeros_like(a)),
        #     (jnp.zeros_like(A_p)),
        #     (A_p,),
        # )[1]
        return phi - phi_0  # - dphi_dA_an_0 - dphi_dA_p_0


### DEFINE THE NGSM2 MODEL ###

key = jax.random.PRNGKey(42)
key1, key2, key3 = jax.random.split(key, num=3)  # define keys for reproducibility

Nvar = 1  # only one additional internal variable
hidden_dims = [8, 8]
elasticity = jm.LinearElasticIsotropic(
    200.0e9, nu=0.3
)  # define the isotropic elastic model. The value wil depend on the normalization of your data. You may expect : \tilde E = E * scale_eps /scale_sig
nn_anelastic = ICNN(1, hidden_dims, key1)  # one input the second main invariant
nn_hardening = ICNN(Nvar, hidden_dims, key2)  # Nvar inputs

free_energy = FreeEnergy(elasticity, nn_anelastic, nn_hardening)
dissipation_potential = DissipationPotential(Nvar + 1, hidden_dims, key3)


class GSM(jm.GeneralizedStandardMaterial):
    def make_internal_state(self):
        return InternalState(Nvar=Nvar)


gsm = GSM(free_energy=free_energy, dissipation_potential=dissipation_potential)
state0 = gsm.init_state()

print(state0)
