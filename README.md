# Krylovable.jl

Everyone loves Krylov(.jl)

Running `fft_versus_pcg_solver.jl` produces:

```julia
julia> include("fft_versus_pcg_solver.jl")
  0.002864 seconds (26 allocations: 6.688 KiB)
 21.955589 seconds (410.61 M allocations: 13.020 GiB, 3.33% gc time)
[ Info: PCG solver iterations: 190
[ Info: Max PCG residual: 1.277806338387602e-9
```

and the figure

<img width="1186" alt="image" src="https://github.com/glwagner/Krylovable.jl/assets/15271942/5b49aaa7-3e49-4c1c-a5d5-20416eb3e090">

The PCG solver seems to perform _very_ poorly and allocates way too much.
I'm not sure why this might be, but it could be an issue with `norm`.

