using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Solvers: FFTBasedPoissonSolver, solve!

using Statistics
using Random
Random.seed!(123)

#####
##### Grid setup
#####

arch = CPU() # try changing to GPU()
Nx = Ny = Nz = 64
topology = (Periodic, Periodic, Periodic)

# Regularly-spaced grid for now:
x = y = z = (0, 2π)

grid = RectilinearGrid(arch; x, y, z, topology, size=(Nx, Ny, Nz))

#####
##### Right-hand side generation: sum of randomly-place Gaussians
#####

δ = 0.1 # Gaussian width
gaussian(ξ, η, ζ) = exp(-(ξ^2 + η^2 + ζ^2) / 2δ^2)
Ngaussians = 17
Ξs = [2π * rand(3) for _ = 1:Ngaussians]

function many_gaussians(ξ, η, ζ)
    val = zero(ξ)
    for Ξ₀ in Ξs
        ξ₀, η₀, ζ₀ = Ξ₀
        val += gaussian(ξ - ξ₀, η - η₀, ζ - ζ₀)
    end
    return val
end

#####
##### FFT-based Poisson solver
#####

# Solve Ax = b
fft_solver = FFTBasedPoissonSolver(grid)

# Note: we use in-place transforms, so the RHS has to be AbstractArray{Complex{T}}.
# So, we first fill up "b" and then copy it into "bc = fft_solver.storage",
# which has the correct type.
b = CenterField(grid)
set!(b, many_gaussians)
parent(b) .-= mean(interior(b))
bc = fft_solver.storage 
bc .= interior(b)

xfft = CenterField(grid)
solve!(xfft, fft_solver, bc)

bc .= interior(b)
@time solve!(xfft, fft_solver, bc)

#####
##### PCG-based Poisson solver
#####

include("preconditioned_conjugate_gradient_poisson_solver.jl")

# Note: this won't work unless the "diagonally-dominant" preconditioner is used
# (which is the default for preconditioned_conjugate_gradient_poisson_solver)
xpcg = CenterField(grid)
pcg_solver = preconditioned_conjugate_gradient_poisson_solver(grid, xpcg; maxiter = 1000)

solve!(xpcg, pcg_solver, b)

# Zero the solution for fairness
parent(xpcg) .= 0
@time solve!(xpcg, pcg_solver, b)

xpcg_krylov, niter = krylov_pcg_poisson_solver(grid, b)
@time krylov_pcg_poisson_solver(grid, b)

xpcg_krylov2 = similar(xpcg_krylov)
krylov_solver = krylov_pcg_poisson_solver2(grid, b)
@time solve!(xpcg_krylov2, krylov_solver, b)

#####
##### Visualize the results, including residuals
#####

using GLMakie

∇²x = CenterField(grid)
compute_laplacian!(∇²x, xfft)
rfft = interior(∇²x) .- interior(b)

∇²x = CenterField(grid)
compute_laplacian!(∇²x, xpcg)
rpcg = interior(∇²x) .- interior(b)

∇²x = CenterField(grid)
compute_laplacian!(∇²x, xpcg_krylov)
rpcg_krylov = interior(∇²x) .- interior(b)

∇²x = CenterField(grid)
compute_laplacian!(∇²x, xpcg_krylov2)
rpcg_krylov2 = interior(∇²x) .- interior(b)

@info "Max FFT residual: " * string(maximum(rfft))
@info "PCG solver iterations: " * string(pcg_solver.iteration)
@info "Max PCG residual: " * string(maximum(rpcg))
@info "PCG -- Krylov.jl solver iterations: " * string(niter)
@info "Max PCG -- Krylov.jl residual: " * string(maximum(rpcg_krylov))
@info "PCG v2 -- Krylov.jl solver iterations: " * string(krylov_solver.workspace.stats.niter)
@info "Max PCG v2 -- Krylov.jl residual: " * string(maximum(rpcg_krylov2))

# Look at yz-slices:
b_cpu    = Array(interior(b, 1, :, :))
xfft_cpu = Array(interior(xfft, 1, :, :))
xpcg_cpu = Array(interior(xpcg, 1, :, :))
xpcg_krylov_cpu = Array(interior(xpcg_krylov, 1, :, :))
xpcg_krylov2_cpu = Array(interior(xpcg_krylov2, 1, :, :))
rfft_cpu = Array(view(rfft, 1, :, :))
rpcg_cpu = Array(view(rpcg, 1, :, :))
rpcg_krylov_cpu = Array(view(rpcg_krylov, 1, :, :))
rpcg_krylov2_cpu = Array(view(rpcg_krylov2, 1, :, :))

fig = Figure(size=(1200, 800))

axb = Axis(fig[1, 1], title="b", aspect=1)
axxfft = Axis(fig[1, 2], title="x (FFT)", aspect=1)
axxpcg = Axis(fig[1, 3], title="x (PCG)", aspect=1)
axxpcg_krylov = Axis(fig[1, 4], title="x (PCG -- Krylov.jl)", aspect=1)
axxpcg_krylov2 = Axis(fig[1, 5], title="x (PCG v2 -- Krylov.jl)", aspect=1)
axrfft = Axis(fig[2, 2], title="r = ∇²x - b (FFT)", aspect=1)
axrpcg = Axis(fig[2, 3], title="r = ∇²x - b (PCG)", aspect=1)
axrpcg_krylov = Axis(fig[2, 4], title="r = ∇²x - b (PCG -- Krylov.jl)", aspect=1)
axrpcg_krylov2 = Axis(fig[2, 5], title="r = ∇²x - b (PCG v2 -- Krylov.jl)", aspect=1)

heatmap!(axb,    b_cpu)
heatmap!(axxfft, xfft_cpu)
heatmap!(axxpcg, xpcg_cpu)
heatmap!(axxpcg_krylov, xpcg_krylov_cpu)
heatmap!(axxpcg_krylov2, xpcg_krylov2_cpu)
heatmap!(axrfft, rfft_cpu)
heatmap!(axrpcg, rpcg_cpu)
heatmap!(axrpcg_krylov, rpcg_krylov_cpu)
heatmap!(axrpcg_krylov2, rpcg_krylov2_cpu)

display(fig)
