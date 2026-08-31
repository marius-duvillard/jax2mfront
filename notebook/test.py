import jaxmat.materials as jm
import equinox as eqx
import lineax as lx
from jaxmat.solvers import NewtonTrustRegion
from optax.tree_utils import tree_add, tree_zeros_like, tree_scale, tree_sub
import jax
import optimistix as optx
from abc import abstractmethod


class GeneralizedStandardMaterialDual(jm.SmallStrainBehavior):
    r"""
    GSM where internal variables evolve by solving the implicit equation with an optax solver.

    dot(alpha) = dPhi_star_dA( A_dis ), with A_dis = - dPsi/dalpha

    Attributes:
        free_energy: eqx.Module, Psi(eps, alpha) -> scalar or tensor
        dual_dissipation: eqx.Module, Phi^*(A_dis) -> scalar (or differentiable scalar-like)
    """

    free_energy: eqx.Module
    r"""Module defining the Helmholtz free energy $\Psi(\beps,\balpha)"""
    dual_dissipation_potential: eqx.Module
    r"""Module defining the dual dissipation pseudo-potential $Phi^*(A_dis)$"""
    minimisation_solver = NewtonTrustRegion(
        rtol=1e-6, atol=1e-6, linear_solver=lx.AutoLinearSolver(well_posed=False)
    )

    @abstractmethod
    def make_internal_state(self):
        """Create internal state variables."""
        pass

    def residual(self, d_isv, args):
        r"""
        Residual of the implicit evolution equation for internal variables.

        Parameters
        ----------
        d_isv : PyTree
            Increment of internal variables Delta alpha.
        args : tuple
            (eps, state, dt)
        """
        eps, state, dt = args
        isv_old = state.internal
        isv = tree_add(isv_old, d_isv)  # alpha_{n+1}
        isv_dot = tree_scale(1.0 / dt, d_isv)

        # Dissipative force A_dis = - dPsi / dalpha
        A_dis = jax.jacfwd(self.free_energy, argnums=1)(eps, isv)
        A_dis = tree_scale(-1, A_dis)

        # Flow rule: dot(alpha) - dPhi*(A_dis) = 0
        flow = jax.grad(self.dual_dissipation_potential)(A_dis)

        return tree_sub(isv_dot, flow)

    def constitutive_update(self, eps, state, dt):
        isv_old = state.internal
        d_isv0 = tree_zeros_like(isv_old)
        args = eps, state, dt

        sol = optx.root_find(
            self.residual,
            solver=self.minimisation_solver,
            y0=d_isv0,
            args=args,
            throw=False,
        )

        d_isv = sol.value
        isv = tree_add(isv_old, d_isv)

        # Stress: sigma = dPsi / deps
        sig = jax.jacfwd(self.free_energy, argnums=0)(eps, isv)

        new_state = state.update(stress=sig, internal=isv)
        return sig, new_state
