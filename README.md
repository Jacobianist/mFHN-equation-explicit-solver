# CUDA Simulation of 1D/2D Three-Component Reaction-Diffusion System (extended FitzHugh-Nagumo)

## Mathematical model

A system of three partial differential equations is simulated:

$$
\frac{\partial u}{\partial t} = D_1 \nabla^2 u + \phi (a u - \alpha u^3 - b v - c w)
$$

$$
\frac{\partial v}{\partial t} = D_2 \nabla^2 v + \phi \varepsilon_2 (u - v)
$$

$$
\frac{\partial w}{\partial t} = D_3 \nabla^2 w + \phi \varepsilon_3 (u - w)
$$

where:

- $u$ --- activator (primary variable, e.g. membrane potential).
- $v, w$ --- inhibitors (recovery variables).
- $D_1, D_2, D_3$ --- diffusion coefficients.
- $\phi, a, b, c, \alpha, \varepsilon_2, \varepsilon_3$ --- parameters of the system kinetics.
