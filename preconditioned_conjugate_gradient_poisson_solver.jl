using Oceananigans.Operators
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: ConjugateGradientSolver, KrylovField
using Oceananigans.Utils: launch!

using KernelAbstractions
import Krylov

import Oceananigans.Solvers: precondition!

struct DiagonallyDominantPreconditioner end

@inline function precondition!(P_r, ::DiagonallyDominantPreconditioner, r, args...)
    fill_halo_regions!(r)
    grid = r.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _MITgcm_precondition!, P_r, grid, r)
    return P_r
end

function LinearAlgebra.mul!(y::KrylovField, P::DiagonallyDominantPreconditioner, x::KrylovField)
    precondition!(y.field, P, x.field)
end

# Helper functions for calculating the coefficients of the "MITgcm" preconditioner
@inline Ax⁻(i, j, k, grid) = Axᶠᶜᶜ(i, j, k, grid)   / Δxᶠᶜᶜ(i, j, k, grid)   / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁻(i, j, k, grid) = Ayᶜᶠᶜ(i, j, k, grid)   / Δyᶜᶠᶜ(i, j, k, grid)   / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁻(i, j, k, grid) = Azᶜᶜᶠ(i, j, k, grid)   / Δzᶜᶜᶠ(i, j, k, grid)   / Vᶜᶜᶜ(i, j, k, grid)
@inline Ax⁺(i, j, k, grid) = Axᶠᶜᶜ(i+1, j, k, grid) / Δxᶠᶜᶜ(i+1, j, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Ay⁺(i, j, k, grid) = Ayᶜᶠᶜ(i, j+1, k, grid) / Δyᶜᶠᶜ(i, j+1, k, grid) / Vᶜᶜᶜ(i, j, k, grid)
@inline Az⁺(i, j, k, grid) = Azᶜᶜᶠ(i, j, k+1, grid) / Δzᶜᶜᶠ(i, j, k+1, grid) / Vᶜᶜᶜ(i, j, k, grid)

@inline Ac(i, j, k, grid) = - (Ax⁻(i, j, k, grid) +
                               Ax⁺(i, j, k, grid) +
                               Ay⁻(i, j, k, grid) +
                               Ay⁺(i, j, k, grid) +
                               Az⁻(i, j, k, grid) +
                               Az⁺(i, j, k, grid))

@inline heuristic_inverse_times_residuals(i, j, k, grid, r) =
    @inbounds 1 / Ac(i, j, k, grid) * (r[i, j, k] - 2 * Ax⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i-1, j, k, grid)) * r[i-1, j, k] -
                                                    2 * Ax⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i+1, j, k, grid)) * r[i+1, j, k] -
                                                    2 * Ay⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j-1, k, grid)) * r[i, j-1, k] -
                                                    2 * Ay⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j+1, k, grid)) * r[i, j+1, k] -
                                                    2 * Az⁻(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k-1, grid)) * r[i, j, k-1] -
                                                    2 * Az⁺(i, j, k, grid) / (Ac(i, j, k, grid) + Ac(i, j, k+1, grid)) * r[i, j, k+1])

@kernel function _MITgcm_precondition!(P_r, grid, r)
    i, j, k = @index(Global, NTuple)
    @inbounds P_r[i, j, k] = heuristic_inverse_times_residuals(i, j, k, grid, r)
end

#=
# Here's the source code implementation of ∇²ᶜᶜᶜ::::
"""
    ∇²ᶜᶜᶜ(i, j, k, grid, c)

Calculate the Laplacian of ``c`` via

```julia
1/V * [δxᶜᵃᵃ(Ax * ∂xᶠᵃᵃ(c)) + δyᵃᶜᵃ(Ay * ∂yᵃᶠᵃ(c)) + δzᵃᵃᶜ(Az * ∂zᵃᵃᶠ(c))]
```

which ends up at the location `ccc`.
"""
@inline function ∇²ᶜᶜᶜ(i, j, k, grid, c)
    return 1/Vᶜᶜᶜ(i, j, k, grid) * (δxᶜᵃᵃ(i, j, k, grid, Ax_∂xᶠᶜᶜ, c) +
                                    δyᵃᶜᵃ(i, j, k, grid, Ay_∂yᶜᶠᶜ, c) +
                                    δzᵃᵃᶜ(i, j, k, grid, Az_∂zᶜᶜᶠ, c))
end

# See here:
#
# https://github.com/CliMA/Oceananigans.jl/blob/1b4736614d3528d6a7b5cad00459bacad580052c/src/Operators/laplacian_operators.jl#L36

=#

@kernel function laplacian!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_laplacian!(∇²ϕ, ϕ)
    fill_halo_regions!(ϕ)
    grid = ϕ.grid
    arch = architecture(grid)
    launch!(arch, grid, :xyz, laplacian!, ∇²ϕ, grid, ϕ)
    return nothing
end

function preconditioned_conjugate_gradient_poisson_solver(grid, rhs=CenterField(grid);
                                                          preconditioner = DiagonallyDominantPreconditioner(),
                                                          reltol = sqrt(eps(eltype(grid))),
                                                          abstol = 0,
                                                          kw...)

    pcg_solver = ConjugateGradientSolver(compute_laplacian!;
                                         template_field = rhs,
                                         reltol,
                                         abstol,
                                         kw...)

    return pcg_solver
end

## Krylov.jl
struct LaplacianOperator
    m::Int
    n::Int
end

Base.size(A::LaplacianOperator) = (A.m, A.n)
Base.eltype(A::LaplacianOperator) = Float64

function LinearAlgebra.mul!(y::KrylovField, A::LaplacianOperator, x::KrylovField)
    compute_laplacian!(y.field, x.field)
end

function krylov_pcg_poisson_solver(grid, rhs=CenterField(grid);
                                   P = DiagonallyDominantPreconditioner(),
                                   reltol = sqrt(eps(eltype(grid))),
                                   abstol = zero(eltype(grid)),
                                   verbose = 0,
                                   kw...)

    n = length(rhs)
    A = LaplacianOperator(n,n)
    b = KrylovField(rhs)
    kc = Krylov.KrylovConstructor(b)
    solver = Krylov.CgSolver(kc)
    # P is not symmetric positive definite!
    # Krylov.cg!(solver, A, b, M=P, verbose=verbose, atol=abstol, rtol=reltol)
    Krylov.cg!(solver, A, b, verbose=verbose, atol=abstol, rtol=reltol)
    return solver.x, solver.stats.niter
end
