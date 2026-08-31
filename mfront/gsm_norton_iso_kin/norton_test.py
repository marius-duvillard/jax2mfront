#!/usr/bin/env python3
"""
Test script for a simple Norton viscoplastic material with isotropic hardening only.
This implements a Norton-type viscoplastic material with:
- Linear elastic isotropic behavior
- Von Mises yield criterion
- Voce isotropic hardening
- Norton flow rule
- No kinematic hardening (H=0)
"""

import jaxmat.materials as jm
from optax.tree_utils import tree_add, tree_zeros_like
from jaxmat.tensors import SymmetricTensor2, dev
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxmat.materials import (
    GenericViscoplasticity,
    LinearElasticIsotropic,
    VoceHardening,
    NortonFlow,
)
import equinox as eqx
import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from optax.tree_utils import tree_add, tree_zeros_like

from jaxmat.state import AbstractState, SmallStrainState
from jaxmat.utils import default_value

from jaxmat.materials.behavior import SmallStrainBehavior
from jaxmat.materials.plastic_surfaces import AbstractPlasticSurface, vonMises
from jaxmat.materials.viscoplastic_flows import (
    AbstractKinematicHardening,
    ArmstrongFrederickHardening,
    NortonFlow,
)
from jaxmat.materials.elastoplasticity import InternalState


class SimpleNorton(SmallStrainBehavior):
    """
    Small-strain viscoplastic constitutive model with generic yield surface, isotropic
    and kinematic hardening and viscoplastic flow rule.
    """

    elasticity: LinearElasticIsotropic
    """Linear isotropic elasticity defined by Young modulus and Poisson ratio."""
    plastic_surface: AbstractPlasticSurface
    """A generic plastic yield surface."""
    yield_stress: eqx.Module
    """Isotropic hardening law controlling the evolution of the yield surface size."""
    viscous_flow: eqx.Module
    """A generic viscoplastic flow rule."""

    def make_internal_state(self):
        return InternalState()

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def sig_eq(sig):
            return self.plastic_surface(sig)

        def eval_stress(deps, dy):
            return sig_old + self.elasticity.C @ (deps - dev(dy.epsp))

        def solve_state(deps, y_old):
            def residual(dy, args):
                y = tree_add(y_old, dy)
                sig = eval_stress(deps, dy)
                yield_criterion = sig_eq(sig) - self.yield_stress(y.p)
                n = self.plastic_surface.normal(sig)
                res = (
                    dy.p - dt * self.viscous_flow(yield_criterion),
                    dy.epsp - n * dy.p,
                )
                return res, y

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(
                residual, self.solver, dy0, has_aux=True, adjoint=self.adjoint
            )
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


def create_simple_norton_material():
    """
    Create a simple Norton viscoplastic material with isotropic hardening only.

    Returns:
        material: GenericViscoplasticity instance
    """
    # Elastic properties (steel-like)
    elasticity = LinearElasticIsotropic(E=210000.0, nu=0.3)

    # Voce isotropic hardening parameters
    # sig0: initial yield stress
    # sigu: saturation stress
    # b: hardening rate
    yield_stress = VoceHardening(sig0=100.0, sigu=200.0, b=10.0)

    # Norton flow parameters
    # K: characteristic stress
    # m: power-law exponent
    viscous_flow = NortonFlow(K=100.0, m=3.0)

    # Kinematic hardening with H=0 to disable it

    # Create the material
    material = SimpleNorton(
        elasticity=elasticity,
        plastic_surface=vonMises(),
        yield_stress=yield_stress,
        viscous_flow=viscous_flow,
    )

    return material


create_simple_norton_material()
