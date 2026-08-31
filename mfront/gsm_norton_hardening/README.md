# Integration of a neural network constitutive law with historical variable into MFront

Our neural network constitutive law is based on a state defined by the strain tensor $\bm \varepsilon^\text{to}$ and the internal variable $\bm \alpha$ and the following equations : 

$$
\bm \sigma = \phi_1(\bm \varepsilon^\text{to}, \bm \alpha) \\
\dot \alpha = \phi_2(\bm \varepsilon^\text{to}, \bm \alpha)
$$

## Implicit discretisation of the law

To integrate this law,into structural calculation, we define a time discretization based on successive time step $\{t_i\}_{1 \leq i \leq I}$.

We use an implicit algorithm, rewriting the evolution equation of internal variables 

$$
\Delta \bm \alpha - \Delta t~\phi_2(\bm \varepsilon^\text{to} + \theta~ \Delta \bm \varepsilon^\text{to}, \bm \alpha + \theta~ \Delta \bm \alpha) = 0\\
= F(\Delta \bm \varepsilon^\text{to}, \Delta \bm \alpha) = 0\\
$$

The increment $\Delta \bm \alpha$ is defined as an implicit function of the strain increment $\Delta \bm \varepsilon^\text{to}$ leading to 

$$
\bm F(\Delta \bm \varepsilon^\text{to}, \Delta \bm \alpha(\Delta \bm \varepsilon^\text{to})) = 0
$$

We recall that the stress $\bm \sigma$ is an explicit function of the integrated variables $\bm \alpha$ at the end of the time step :
$$
\bm \sigma \left( \left. \bm \varepsilon^\text{to}\right|_{t+\Delta t}, \left.\bm \alpha\right|_{t+\Delta t}\right) = \phi_1(\left. \bm \varepsilon^\text{to}\right|_{t+\Delta t}, \left.\bm \alpha\right|_{t+\Delta t})
$$
## Computation of the consistent tangent operator

The consistent operator is defined as 

$$
\frac{\mathrm{d} \bm \sigma}{\mathrm{d} \Delta \bm \varepsilon^\text{to}} = \frac{\partial \bm \phi_1}{\partial  \Delta \bm \varepsilon^\text{to}} + \frac{\partial \bm \phi_1}{\partial  \Delta \bm \alpha}\frac{\mathrm{d} \Delta \bm \alpha}{\mathrm{d}  \Delta \bm \varepsilon^\text{to}}
$$

The first term could directly be determined knowing $\partial_1 \bm \phi_1$ using auto-differentiation if the neural network $\bm \phi_1$ and similarly for $\frac{\partial \bm \phi_1}{\partial  \Delta \bm \alpha}$ with $\partial_2 \bm \phi_1$.

To determined $\frac{\partial \Delta \bm \alpha}{\partial  \Delta \bm \varepsilon^\text{to}}$, we will use the implicit theorem over the equation $F$.

Let's differentiate 

$$
\frac{\mathrm{d} \bm F}{\mathrm{d} \Delta \bm \varepsilon^\text{to}} = \frac{\partial \bm F}{\partial  \Delta \bm \varepsilon^\text{to}} + \frac{\partial \bm F}{\partial  \Delta \bm \alpha}\frac{\mathrm{d} \Delta \bm \alpha}{\mathrm{d} \Delta \bm \varepsilon^\text{to}} = \bm 0 
$$

By named the jacobian $J = \frac{\partial \bm F}{\partial  \Delta \bm \alpha} = \bm I - \Delta t~ \frac{\partial \bm \phi_2}{\partial \Delta \bm \alpha}$, we obtain by inversion 

$$
\frac{\mathrm{d} \Delta \bm \alpha}{\mathrm{d}  \Delta \bm \varepsilon^\text{to}} = - J^{-1} \frac{\partial \bm F}{\partial  \Delta \bm \varepsilon^\text{to}}.  
$$

This is express as

$$
\frac{\mathrm{d} \Delta \bm \alpha}{\mathrm{d}  \Delta \bm \varepsilon^\text{to}} = - J^{-1} \frac{\partial \bm F}{\partial  \Delta \bm \varepsilon^\text{to}}.  
$$

Leading to the final expression of the operator tangent

$$
\frac{\mathrm{d} \bm \sigma}{\mathrm{d} \Delta \bm \varepsilon^\text{to}} = \frac{\partial \bm \phi_1}{\partial  \Delta \bm \varepsilon^\text{to}} - \theta~\frac{\partial \bm \phi_1}{\partial  \Delta \bm \alpha}J^{-1} \frac{\partial \bm F}{\partial  \Delta \bm \varepsilon^\text{to}}
$$

Which is rewrite with the derivatives of $\bm \phi_2$

$$
\frac{\mathrm{d} \bm \sigma}{\mathrm{d} \Delta \bm \varepsilon^\text{to}} = \frac{\partial \bm \phi_1}{\partial  \Delta \bm \varepsilon^\text{to}} + \theta~\Delta t \frac{\partial \bm \phi_1}{\partial  \Delta \bm \alpha}J^{-1} \frac{\partial \bm \phi_2}{\partial  \Delta \bm \varepsilon^\text{to}}.
$$

