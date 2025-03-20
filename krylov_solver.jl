using Krylov
using LinearAlgebra
import Oceananigans.Architectures: architecture
import Oceananigans.Fields: AbstractField
import LinearAlgebra: mul!

mutable struct KrylovSolver{A,G,L,S,P,T}
    architecture :: A
    grid :: G
    op :: L
    workspace :: S
    krylov_solver :: Symbol
    preconditioner :: P
    abstol::T
    reltol::T
    maxiter::Int
    maxtime::Float64
end

struct KrylovOperator{F}
    m::Int
    n::Int
    fun::F
end

Base.size(A::KrylovOperator) = (A.m, A.n)
Base.eltype(A::KrylovOperator) = Float64
LinearAlgebra.mul!(y, A::KrylovOperator, x) = A.fun(y.field, x.field)

architecture(solver::KrylovSolver) = solver.architecture
Base.summary(solver::KrylovSolver) = "KrylovSolver"

function KrylovSolver(linear_operator;
                      template_field::AbstractField,
                      maxiter = prod(size(template_field)),
                      maxtime = Inf,
                      reltol = sqrt(eps(eltype(template_field.grid))),
                      abstol = zero(eltype(template_field.grid)),
                      preconditioner = nothing,
                      krylov_solver::Symbol = :cg)

    arch = architecture(template_field)
    grid = template_field.grid
    FT = eltype(grid)

    m = n = length(template_field)
    op = KrylovOperator(m, n, linear_operator)

    kf = KrylovField(template_field)
    kc = Krylov.KrylovConstructor(kf)
    workspace = eval(Krylov.KRYLOV_SOLVERS[krylov_solver])(kc)

    return KrylovSolver(arch,
                        grid,
                        op,
                        workspace,
                        krylov_solver,
                        I,
                        FT(abstol),
                        FT(reltol),
                        maxiter,
                        maxtime)

end

function Oceananigans.solve!(x, solver::KrylovSolver, b, args...; kwargs...)
    Krylov.solve!(solver.workspace, solver.op, KrylovField(b); M=solver.preconditioner,
                  atol=solver.abstol, rtol=solver.reltol, itmax=solver.maxiter, timemax=solver.maxtime, kwargs...)
    copyto!(x, solver.workspace.x)
    return x
end
