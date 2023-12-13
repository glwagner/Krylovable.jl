using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!

using Statistics
using Random
Random.seed!(123)

arch = CPU()
Nx = Ny = Nz = 32
topology = (Periodic, Periodic, Periodic)

# Regularly-spaced grid for now:
x = y = z = (0, 2π)

grid = RectilinearGrid(arch; x, y, z, topology, size=(Nx, Ny, Nz))

# Generate an interesting right-hand side
δ = 0.1
ε = 0.5
gaussian(ξ, η, ζ) = exp(-(ξ^2 + η^2 + ζ^2) / 2δ^2)
Ngaussians = 17
Ξ = [ε/2 .+ (2π - ε) * rand(3) for _ = 1:Ngaussians]

function b_func(ξ, η, ζ)
    val = zero(ξ)
    for Ξ₀ in Ξ
        ξ₀, η₀, ζ₀ = Ξ₀
        val += gaussian(ξ - ξ₀, η - η₀, ζ - ζ₀)
    end
    return val
end

# Solve Ax = b
fft_solver = FFTBasedPoissonSolver(grid)

# Note: we use in-place transforms, so the RHS has to be AbstractArray{Complex{T}}.
# So, we first fill up "b" and then copy it into "bc = fft_solver.storage",
# which has the correct type.
b = CenterField(grid)
set!(b, b_func)
parent(b) .-= mean(interior(b))
bc = fft_solver.storage 
bc .= interior(b)

xfft = CenterField(grid)
solve!(xfft, fft_solver, bc)
@time solve!(xfft, fft_solver, bc)

include("preconditioned_conjugate_gradient_poisson_solver.jl")
xpcg = CenterField(grid)
pcg_solver = preconditioned_conjugate_gradient_poisson_solver(grid, xpcg;
                                                              preconditioner = nothing,
                                                              maxiter = 10)
solve!(xpcg, pcg_solver, b)
parent(xpcg) .= parent(xfft)
@time solve!(xpcg, pcg_solver, b)

using GLMakie

b_cpu    = Array(interior(b, 1, :, :))
xfft_cpu = Array(interior(xfft, 1, :, :))
xpcg_cpu = Array(interior(xpcg, 1, :, :))

fig = Figure(size=(1200, 400))
axb = Axis(fig[1, 1], title="b")
axxfft = Axis(fig[1, 2], title="x (FFT)")
axxpcg = Axis(fig[1, 3], title="x (PCG)")
heatmap!(axb,    b_cpu)
heatmap!(axxfft, xfft_cpu)
heatmap!(axxpcg, xpcg_cpu)
display(fig)

