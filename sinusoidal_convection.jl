using Oceananigans
using Oceananigans.Units
using Printf

arch = CPU()
Nx = Ny = Nz = 128
x = y = (0, 512)
z = (0, 256)

grid = RectilinearGrid(arch; size=(Nx, 1, Nz), x, y, z)

N² = 1e-5
Jb = 1e-7
ϵ = 0.5
Lx = grid.Lx
@inline sinusoidal_flux(x, y, t, p) = p.Jb * (1 - p.ϵ * cos(2π * x / p.Lx))
bottom_buoyancy_flux = FluxBoundaryCondition(sinusoidal_flux, parameters=(; Jb, ϵ, Lx))
b_bcs = FieldBoundaryConditions(bottom = bottom_buoyancy_flux)

model = NonhydrostaticModel(; grid,
                            advection = WENO(),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            boundary_conditions = (; b=b_bcs))

bi(x, y, z) = N² * z + 1e-9 * randn()
set!(model, b=bi)

simulation = Simulation(model, Δt=1.0, stop_time=24hours)
conjure_time_step_wizard!(simulation, cfl=0.5)

function progress(sim)
    w = sim.model.velocities.w
    msg = @sprintf("Iter: %d, time: %s, dt: %s, max|w|: %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, w))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(10))

prefix = @sprintf("heterogeneous_convection_ep%d_N%d", ϵ, Nx)

u, v, w = model.velocities
b = model.tracers.b
B = Average(b, dims=(1, 2))
w² = Average(w^2, dims=(1, 2))
averages = (; B, w²)
avg_ow = JLD2OutputWriter(model, averages,
                          filename = prefix * "_averages.jld2",
                          schedule = TimeInterval(1hour),
                          overwrite_existing = true)

simulation.output_writers[:avg] = avg_ow

outputs = merge(model.velocities, model.tracers)
slices = JLD2OutputWriter(model, outputs,
                          filename = prefix * "_xz.jld2",
                          indices = (:, 1, :),
                          schedule = TimeInterval(1hour),
                          overwrite_existing = true)

simulation.output_writers[:slices] = slices

run!(simulation)

wt = FieldTimeSeries("xz_slices.jld2", "w")
Bt = FieldTimeSeries("averages.jld2", "B")
Nt = length(Bt)

n = Nt
Bn = Bt[n]
wn = wt[n]
z = znodes(Bt)

fig = Figure()
axb = Axis(fig[1, 1])
axw = Axis(fig[1, 2])
lines!(axb, Bn) #, z)
heatmap!(axw, wn) #, z)

