using Printf
using Statistics
using Oceananigans
using Oceananigans.Grids: with_number_type
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, fft_poisson_solver

N = 16
H, L = 100, 100
x = y = (-L/2, L/2)
z = (-H, 0)

grid = RectilinearGrid(size=(N, N, N); x, y, z, halo=(3, 3, 3), topology=(Periodic, Periodic, Bounded))

# reduced_precision_grid = with_number_type(Float32, grid.underlying_grid)
reduced_precision_grid = with_number_type(Float32, grid)
preconditioner = fft_poisson_solver(reduced_precision_grid)
# pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter=1000, regularization=1/N^3)
pressure_solver = ConjugateGradientPoissonSolver(grid; preconditioner, maxiter=1000)

model = NonhydrostaticModel(; grid, pressure_solver)

ui(x, y, z) = randn()
set!(model, u=ui, v=ui, w=ui)

simulation = Simulation(model; Δt=0.1, stop_iteration=100)
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
    JLD2OutputWriter(model, model.velocities; filename="test.jld2", schedule=IterationInterval(10), overwrite_existing=true)

run!(simulation)
