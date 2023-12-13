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

# I can't get the following to work... not sure why.
xpcg = CenterField(grid)
pcg_solver = preconditioned_conjugate_gradient_poisson_solver(grid, xpcg; maxiter = 1000)

parent(xpcg) .= 0 #arent(xfft)
solve!(xpcg, pcg_solver, b)

parent(xpcg) .= 0 #parent(xfft)
@time solve!(xpcg, pcg_solver, b)

#####
##### Visualize the results, including residual
#####

using GLMakie

∇²x = CenterField(grid)
compute_laplacian!(∇²x, xfft)
rfft = interior(∇²x) .- interior(b)

compute_laplacian!(∇²x, xpcg)
rpcg = interior(∇²x) .- interior(b)

@info "PCG solver iterations: " * string(pcg_solver.iteration)
@info "Max PCG residual: " * string(maximum(rpcg))

b_cpu    = Array(interior(b, 1, :, :))
xfft_cpu = Array(interior(xfft, 1, :, :))
xpcg_cpu = Array(interior(xpcg, 1, :, :))
rfft_cpu = Array(view(rfft, 1, :, :))
rpcg_cpu = Array(view(rpcg, 1, :, :))

fig = Figure(size=(1200, 800))

axb = Axis(fig[1, 1], title="b", aspect=1)

axxfft = Axis(fig[1, 2], title="x (FFT)", aspect=1)
axxpcg = Axis(fig[1, 3], title="x (PCG)", aspect=1)
axrfft = Axis(fig[2, 2], title="r = ∇²x - b (FFT)", aspect=1)
axrpcg = Axis(fig[2, 3], title="r = ∇²x - b (PCG)", aspect=1)

heatmap!(axb,    b_cpu)
heatmap!(axxfft, xfft_cpu)
heatmap!(axxpcg, xpcg_cpu)
heatmap!(axrfft, rfft_cpu)
heatmap!(axrpcg, rpcg_cpu)

display(fig)

