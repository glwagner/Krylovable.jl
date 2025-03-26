using Printf
using Statistics
using Oceananigans
using Oceananigans.Grids: with_number_type
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver
using Oceananigans.Simulations: NaNChecker
using Oceananigans.Utils: prettytime

N = 16
h, w = 50, 20
H, L = 100, 100
x = y = (-L/2, L/2)
z = (-H, 0)

grid = RectilinearGrid(size=(N, N, N); x, y, z, halo=(2, 2, 2), topology=(Bounded, Periodic, Bounded))

mount(x, y=0) = h * exp(-(x^2 + y^2) / 2w^2)
bottom(x, y=0) = -H + mount(x, y)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

prescribed_flow = OpenBoundaryCondition(0.01)
extrapolation_bc = FlatExtrapolationOpenBoundaryCondition()
u_bcs = FieldBoundaryConditions(west = prescribed_flow,
                                east = extrapolation_bc)
                                #east = prescribed_flow)

boundary_conditions = (; u=u_bcs)
reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
preconditioner = fft_poisson_solver(reduced_precision_grid)
# pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter=1000, regularization=1/N^3)
pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter=1000)

model = NonhydrostaticModel(; grid, boundary_conditions, pressure_solver)
simulation = Simulation(model; Δt=0.1, stop_iteration=1000)
conjure_time_step_wizard!(simulation, cfl=0.5)

u, v, w = model.velocities
δ = ∂x(u) + ∂y(v) + ∂z(w)

function progress(sim)
    model = sim.model
    u, v, w = model.velocities
    @printf("Iter: %d, time: %.1f, Δt: %.2e, max|δ|: %.2e",
            iteration(sim), time(sim), sim.Δt, maximum(abs, δ))

    r = model.pressure_solver.conjugate_gradient_solver.residual
    @printf(", solver iterations: %d, max|r|: %.2e\n",
            iteration(model.pressure_solver), maximum(abs, r))
end

add_callback!(simulation, progress)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, model.velocities; filename="3831.jld2", schedule=IterationInterval(10), overwrite_existing=true)

run!(simulation)

using GLMakie

ds = FieldDataset("3831.jld2")
fig = Figure(size=(1000, 500))

n = Observable(1)
times = ds["u"].times
title = @lift @sprintf("time = %s", prettytime(times[$n]))

Nx, Ny, Nz = size(grid)
j = round(Int, Ny/2)
k = round(Int, Nz/2)
u_surface = @lift view(ds["u"][$n], :, :, k)
u_slice = @lift view(ds["u"][$n], :, j, :)

ax1 = Axis(fig[1, 1]; title = "u (xy)", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, u_surface, colorrange=(-0.01, 0.01), colormap=:balance)
Colorbar(fig[1, 2], hm1, label="m/s")

ax2 = Axis(fig[1, 3]; title = "u (xz)", xlabel="x", ylabel="z")
hm2 = heatmap!(ax2, u_slice, colorrange=(-0.01, 0.01), colormap=:balance)
Colorbar(fig[1, 4], hm2, label="m/s")

fig[0, :] = Label(fig, title)

record(fig, "3831.mp4", 1:length(times), framerate=10) do i
    n[] = i
end
