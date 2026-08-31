from abc import abstractmethod

import equinox as eqx
import jax
import lineax as lx
import optimistix as optx
from optax.tree_utils import tree_add, tree_zeros_like, tree_scale, tree_sub
from optax.tree_utils import tree_add, tree_scale, tree_zeros_like

import jaxmat.materials as jm
from jaxmat.solvers import NewtonTrustRegion
import jaxmat.materials as jm
import lineax as lx


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

    def residual_mixed(self, unknowns, args):
        """
        unknowns = (d_isv, eps_yy)
        depsilon = eps_xx
        args = (eps, state, dt)
        """

        d_isv, d_eps = unknowns
        eps_xx, state, dt = args

        # --- internal variables ---
        isv_old = state.internal
        eps_old = state.strain.array

        isv = tree_add(isv_old, d_isv)
        # isv_dot = tree_scale(1.0 / dt, d_isv)

        # update strain (yy, zz unknown)
        eps = eps_old
        eps = eps.at[0].set(eps_xx)
        eps = eps.at[1].set(eps_old[1] + d_eps[0])
        eps = eps.at[2].set(eps_old[2] + d_eps[1])

        eps = type(state.strain)(array=eps)

        # --- thermodynamic forces ---
        A_dis = jax.jacfwd(self.free_energy, argnums=1)(eps, isv)
        A_dis = tree_scale(-1.0, A_dis)

        # --- flow rule ---
        flow = jax.grad(self.dual_dissipation_potential)(A_dis)
        res_isv = tree_sub(d_isv, tree_scale(dt, flow))

        # --- stresses ---
        sigma = jax.grad(self.free_energy, argnums=0)(eps, isv)

        res_sig_y = sigma.tensor[1, 1]
        res_sig_z = sigma.tensor[2, 2]

        return res_isv, jax.numpy.array([res_sig_y, res_sig_z])

    def constitutive_update_mixed_control(self, eps_xx, state, dt):
        isv_old = state.internal
        d_isv0 = tree_zeros_like(isv_old)
        d_eps0 = jax.numpy.zeros(2)

        args = eps_xx, state, dt

        sol = optx.root_find(
            self.residual_mixed,
            solver=self.minimisation_solver,
            y0=(d_isv0, d_eps0),
            args=args,
            throw=False,
        )

        d_isv, d_eps = sol.value
        eps = state.strain.array
        eps = eps.at[0].set(eps_xx)
        eps = eps.at[1].set(eps[1] + d_eps[0])
        eps = eps.at[2].set(eps[2] + d_eps[1])
        eps = type(state.strain)(array=eps)

        isv = tree_add(isv_old, d_isv)

        sig = jax.jacfwd(self.free_energy, argnums=0)(eps, isv)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GeneralizedStandardMaterialDualParam(jm.SmallStrainBehavior):
    r"""
    GSM where internal variables evolve by solving the implicit equation with an optax solver. Consider parameters as input.

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
        eps, state, dt, theta = args
        isv_old = state.internal
        isv = tree_add(isv_old, d_isv)  # alpha_{n+1}
        isv_dot = tree_scale(1.0 / dt, d_isv)

        # Dissipative force A_dis = - dPsi / dalpha
        A_dis = jax.jacfwd(self.free_energy, argnums=1)(eps, isv, theta)
        A_dis = tree_scale(-1, A_dis)

        # Flow rule: dot(alpha) - dPhi*(A_dis) = 0
        flow = jax.grad(self.dual_dissipation_potential)(A_dis, theta)

        return tree_sub(isv_dot, flow)

    def constitutive_update(self, eps, state, dt, theta):
        isv_old = state.internal
        d_isv0 = tree_zeros_like(isv_old)
        args = eps, state, dt, theta

        sol = optx.root_find(
            self.residual,
            solver=optx.Newton(rtol=1e-8, atol=1e-8),
            y0=d_isv0,
            args=args,
            throw=False,
        )

        d_isv = sol.value
        isv = tree_add(isv_old, d_isv)

        # Stress: sigma = dPsi / deps
        sig = jax.jacfwd(self.free_energy, argnums=0)(eps, isv, theta)

        new_state = state.update(stress=sig, internal=isv)
        return sig, new_state
