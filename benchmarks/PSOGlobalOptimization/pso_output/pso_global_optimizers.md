---
author: "Utkarsh, Chris Rackauckas"
title: "PSO Global Optimizer Benchmarks"
---


This benchmark evaluates Particle Swarm Optimization (PSO) variants from
[ParallelParticleSwarms.jl](https://github.com/SciML/ParallelParticleSwarms.jl) against
established global optimizers on the
[BlackBoxOptimizationBenchmarking.jl](https://github.com/jonathanBieler/BlackBoxOptimizationBenchmarking.jl)
suite (v2 API), using the [Optimization.jl](https://github.com/SciML/Optimization.jl) interface.

## Setup

```julia
using Random
Random.seed!(42)

using BlackBoxOptimizationBenchmarking, CairoMakie, Optimization, Memoize, Statistics
import BlackBoxOptimizationBenchmarking: Chain, BenchmarkSetup, BenchmarkResults,
    BBOBFunction, FunctionCallsCounter, solve_problem, pinit, compute_CI
const BBOB = BlackBoxOptimizationBenchmarking

using OptimizationBBO, OptimizationOptimJL, OptimizationEvolutionary, OptimizationNLopt
using OptimizationMetaheuristics, OptimizationSciPy

using ParallelParticleSwarms
using ForwardDiff
using KernelAbstractions
using CUDA
using StaticArrays, LinearAlgebra

const PSOKernel     = ParallelParticleSwarms.ParallelPSOKernel
const SyncPSOKernel = ParallelParticleSwarms.ParallelSyncPSOKernel
const SerPSO        = ParallelParticleSwarms.SerialPSO
const HPso          = ParallelParticleSwarms.HybridPSO

const BACKEND = CUDABackend()
```

```
CUDA.CUDAKernels.CUDABackend(false, false)
```



```julia
const MARKERS = [:circle, :diamond, :utriangle, :rect, :star5, :dtriangle, :pentagon,
    :hexagon, :cross, :xcross, :star4, :star8, :vline, :hline, :+, :x]
const LINESTYLES = [:solid, :dash, :dot, :dashdot, :dashdotdot]

const PLOT_WIDTH = 1000
const PLOT_HEIGHT = 400
const STROKEWIDTH = 2.5

function plot_success_curves(x_series, y_series, series_labels; xlabel, ylabel = "Success rate",
        xlim = nothing, ylim = (0, 1), size = (PLOT_WIDTH, PLOT_HEIGHT))
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; xlabel, ylabel, xscale = log10, spinewidth = STROKEWIDTH)
    if xlim !== nothing
        xlims!(ax; low = xlim[1], high = xlim[2])
    end
    ylims!(ax; low = ylim[1], high = ylim[2])

    n = length(series_labels)
    colors = cgrad(:seaborn_bright, n; categorical = true)
    handles, leg_labels = [], String[]
    for j in 1:n
        marker = MARKERS[mod1(j, length(MARKERS))]
        style = LINESTYLES[mod1(j, length(LINESTYLES))]
        l = lines!(ax, x_series[j], y_series[j]; color = colors[j], linewidth = 2.5,
            linestyle = style)
        sc = scatter!(ax, x_series[j], y_series[j]; color = colors[j], markersize = 12,
            marker = marker)
        push!(handles, [l, sc])
        push!(leg_labels, series_labels[j])
    end
    Legend(fig[1, 2], handles, leg_labels; framevisible = true, labelsize = 12)
    fig
end

function solve_problem_baseline(optimizer::Union{Chain, BenchmarkSetup}, f, D::Int,
        run_length::Int)
    solve_problem(optimizer, f, D, run_length)
end

function benchmark_time_to_success(
    optimizer::Union{Chain, BenchmarkSetup}, funcs::Vector{<:BBOBFunction};
    Ntrials::Int = 15, dimension::Int = 3, Δf::Real = 1e-6, max_run_length::Int = 100_000
)
    all_times = Float64[]
    for f in funcs
        for _ in 1:Ntrials
            t0  = time()
            sol = solve_problem_baseline(optimizer, f, dimension, max_run_length)
            elapsed = time() - t0
            push!(all_times, sol.objective < Δf + f.f_opt ? elapsed : Inf)
        end
    end
    return all_times
end

benchmark_time_to_success(optimizer, funcs::Vector{<:BBOBFunction}; kwargs...) =
    benchmark_time_to_success(BenchmarkSetup(optimizer), funcs; kwargs...)

function success_rate_cdf(all_times::Vector{Float64}, time_thresholds::AbstractVector{Float64})
    N = length(all_times)
    return [count(x -> x <= t, all_times) / N for t in time_thresholds]
end
```

```
success_rate_cdf (generic function with 1 method)
```



```julia
_to_f64(x::Real) = Float64(x)
_to_f64(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))
_to_f64(x)       = Float64(x[])

_value(x::Real) = x
_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_penalty(x) = eltype(x) <: ForwardDiff.Dual ? zero(first(x)) + 1.0f10 : 1.0f10

function pso_objective(f::BBOBFunction, x)
    any(xi -> !isfinite(_value(xi)) || abs(_value(xi)) > 15, x) && return _penalty(x)
    y = f(x)
    y isa ForwardDiff.Dual ? y : Float32(y)
end

function _pso_problem(f::BBOBFunction, D::Int; x0 = nothing)
    optf = OptimizationFunction{false}((x, p) -> pso_objective(f, x), Optimization.SciMLBase.NoAD())
    lb   = SVector{D, Float32}(ntuple(_ -> -5.0f0, Val(D)))
    ub   = SVector{D, Float32}(ntuple(_ ->  5.0f0, Val(D)))
    x0   = x0 === nothing ?
        SVector{D, Float32}(ntuple(_ -> -5.0f0 + rand(Float32) * 10.0f0, Val(D))) :
        SVector{D, Float32}(x0)
    OptimizationProblem{false}(optf, x0, nothing; lb, ub)
end

function pso_solve(opt, f::BBOBFunction, D::Int, maxiters::Int;
        local_maxiters::Int = 50, x0 = nothing)
    prob = _pso_problem(f, D; x0)
    if opt isa HPso
        solve(prob, opt; maxiters, local_maxiters, abstol = 1.0f-8, reltol = 1.0f-8)
    else
        solve(prob, opt; maxiters)
    end
end

function _extract_u(sol, D)
    u = sol.u
    u isa SVector && return u
    u isa AbstractVector && return SVector{D}(u)
    u[]
end

function pso_benchmark(opt, funcs, run_length;
        Ntrials = 15, dimension = 3, local_maxiters = 50, Δf = 1e-6, CI_quantile = 0.25,
        n_particles = 1)
    Nf = length(funcs); Nr = length(run_length)
    success = zeros(Float64, Nf, Nr)
    dist    = zeros(Float64, Nf, Nr)
    fmin    = zeros(Float64, Nf, Nr)
    t0 = time()
    for (fi, f) in enumerate(funcs)
        xopt = SVector{dimension, Float32}(f.x_opt[1:dimension])
        for (ri, rl) in enumerate(run_length)
            hits = 0; dsum = 0.0; fsum = 0.0
            for _ in 1:Ntrials
                sol = pso_solve(opt, f, dimension, rl; local_maxiters)
                u    = _extract_u(sol, dimension)
                fval = _to_f64(sol.objective)
                hits += abs(fval - f.f_opt) < Δf ? 1 : 0
                dsum += Float64(norm(u .- xopt))
                fsum += fval - f.f_opt
            end
            success[fi, ri] = hits / Ntrials
            dist[fi, ri]    = dsum / Ntrials
            fmin[fi, ri]    = fsum / Ntrials
        end
    end
    elapsed = time() - t0
    Neff = Ntrials * Nf
    sr   = vec(mean(success, dims = 1))
    sc   = vec(sum(success .* Ntrials, dims = 1)) .|> round .|> Int
    ci   = BBOB.compute_CI(sr, Neff, CI_quantile)
    BenchmarkResults(
        run_length                = collect(run_length),
        success_count             = sc,
        success_rate              = sr,
        success_rate_qlow         = ci.success_rate_qlow,
        success_rate_qhigh        = ci.success_rate_qhigh,
        distance_to_minimizer     = vec(mean(dist, dims = 1)),
        minimum                   = vec(mean(fmin, dims = 1)),
        runtime                   = elapsed,
        Neffective                = Neff,
        callcount                 = Float64.(run_length) .* n_particles,
        success_rate_per_function = [success[fi, end] for fi in 1:Nf],
    )
end

function pso_tts(opt, funcs; Ntrials = 15, dimension = 3, Δf = 1e-6,
        local_maxiters = 50, max_run_length = 100_000)
    all_times = Float64[]
    D = dimension
    for f in funcs
        for _ in 1:Ntrials
            x0   = SVector{D, Float32}(ntuple(_ -> -5.0f0 + rand(Float32) * 10.0f0, Val(D)))
            prob = _pso_problem(f, D; x0)
            t0 = time()
            sol = if opt isa HPso
                solve(prob, opt; maxiters = max_run_length,
                      local_maxiters, abstol = 1.0f-8, reltol = 1.0f-8)
            else
                solve(prob, opt; maxiters = max_run_length)
            end
            elapsed = time() - t0
            fval = _to_f64(sol.objective)
            push!(all_times, abs(fval - f.f_opt) < Δf ? elapsed : Inf)
        end
    end
    all_times
end
```

```
pso_tts (generic function with 1 method)
```



```julia
chain = (t; isboxed = false) -> Chain(
    BenchmarkSetup(t, isboxed = isboxed),
    BenchmarkSetup(NelderMead(), isboxed = false),
    0.9)

dimension      = 3
test_functions = filter(f -> nameof(f.f) !== :f7, BBOB.bbob_suite(Val(dimension)))
run_length     = round.(Int, 10 .^ LinRange(1, 4, 15))
Ntrials        = 15
num_particles  = 5_000

const SUCCESS_Δf = 1e-6

PSO_KEYS = Set(["SerialPSO", "PSOKernel", "SyncPSOKernel", "HybridPSO_LBFGS"])

setup = Dict(
    "NelderMead"                       => NelderMead(),
    "NLopt.GN_CRS2_LM()"               => chain(NLopt.GN_CRS2_LM(), isboxed = true),
    "NLopt.GN_DIRECT()"                => chain(NLopt.GN_DIRECT(), isboxed = true),
    "NLopt.GN_ESCH()"                  => chain(NLopt.GN_ESCH(), isboxed = true),
    "OptimizationEvolutionary.GA()"    => chain(OptimizationEvolutionary.GA(), isboxed = true),
    "OptimizationEvolutionary.DE()"    => chain(OptimizationEvolutionary.DE(), isboxed = true),
    "OptimizationEvolutionary.ES()"    => chain(OptimizationEvolutionary.ES(), isboxed = true),
    "Optim.SAMIN"                      => chain(SAMIN(verbosity = 0), isboxed = true),
    "BBO_adaptive_de_rand_1_bin"       => chain(BBO_adaptive_de_rand_1_bin(), isboxed = true),
    "BBO_de_rand_2_bin"                => chain(BBO_de_rand_2_bin(), isboxed = true),
    "OptimizationMetaheuristics.ECA"   => chain(OptimizationMetaheuristics.ECA(), isboxed = true),
    "OptimizationMetaheuristics.DE"    => chain(OptimizationMetaheuristics.DE(), isboxed = true),
    "ScipyDifferentialEvolution"       => chain(ScipyDifferentialEvolution(), isboxed = true),
    "SerialPSO"        => SerPSO(512),
    "PSOKernel"        => PSOKernel(num_particles; backend = BACKEND, global_update = true),
    "SyncPSOKernel"    => SyncPSOKernel(num_particles; backend = BACKEND),
    "HybridPSO_LBFGS"  => HPso(pso = SyncPSOKernel(num_particles; backend = BACKEND); backend = BACKEND),
)

@memoize run_bench(algo) = algo in PSO_KEYS ?
    pso_benchmark(setup[algo], test_functions, run_length;
        Ntrials, dimension, Δf = SUCCESS_Δf, n_particles = num_particles) :
    BBOB.benchmark(setup[algo], test_functions, run_length;
        Ntrials, Δf = SUCCESS_Δf)

@memoize run_tts(algo) = algo in PSO_KEYS ?
    pso_tts(setup[algo], test_functions;
        Ntrials, dimension, Δf = SUCCESS_Δf, max_run_length = 10_000) :
    benchmark_time_to_success(setup[algo], test_functions;
        Ntrials, dimension, Δf = SUCCESS_Δf, max_run_length = 10_000)
```

```
run_tts (generic function with 1 method)
```





## Test all (iterations)

```julia
labels  = collect(keys(setup))
results = Array{BBOB.BenchmarkResults}(undef, length(setup))

for (i, algo) in enumerate(labels)
    algo in PSO_KEYS || continue
    results[i] = run_bench(algo)
    @info "PSO success rate" algo success_rate = round(results[i].success_rate[end], digits = 3)
end

for (i, algo) in enumerate(labels)
    algo in PSO_KEYS && continue
    results[i] = run_bench(algo)
end

results
```

```
17-element Vector{BlackBoxOptimizationBenchmarking.BenchmarkResults}:
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.00350877, 0.0140351, 0.0105263, 0.0175439, 0.0315789, 0.0
350877, 0.045614, 0.0526316, 0.0877193, 0.17193, 0.361404, 0.519298, 0.5578
95, 0.54386, 0.578947]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0210526, 0.0315789, 0.0526316, 0.0526316, 0.119298, 0.217
544, 0.410526, 0.529825, 0.564912, 0.554386, 0.568421, 0.561404, 0.557895, 
0.57193, 0.550877]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0, 0.0, 0.0, 0.00350877, 0.0350877, 0.0385965, 0.0491228,
 0.0526316, 0.108772, 0.231579, 0.508772, 0.6, 0.617544, 0.652632, 0.796491
]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0140351, 0.0105263, 0.00701754, 0.0175439, 0.0315789, 0.0
491228, 0.0526316, 0.0526316, 0.108772, 0.259649, 0.515789, 0.649123, 0.722
807, 0.740351, 0.803509]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0, 0.0526316, 0.0526316, 0.0526316, 0.0526316, 0.0526316,
 0.0526316, 0.0526316, 0.157895, 0.315789, 0.473684, 0.684211, 0.684211, 0.
684211, 0.789474]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.00350877, 0.00350877, 0.0105263, 0.0105263, 0.0175439, 0.
0350877, 0.0526316, 0.0526316, 0.105263, 0.164912, 0.382456, 0.519298, 0.56
1404, 0.592982, 0.631579]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0315789, 0.0526316, 0.0526316, 0.0526316, 0.105263, 0.105
263, 0.105263, 0.105263, 0.105263, 0.105263, 0.368421, 0.473684, 0.526316, 
0.526316, 0.578947]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0, 0.0, 0.00350877, 0.0, 0.00701754, 0.0175439, 0.0315789
, 0.0491228, 0.0596491, 0.136842, 0.308772, 0.480702, 0.536842, 0.554386, 0
.557895]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0526316, 0.0526316, 0.0842105, 0.196491, 0.54386, 0.72982
5, 0.782456, 0.82807, 0.852632, 0.85614, 0.873684, 0.880702, 0.880702, 0.87
7193, 0.873684]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.410526, 0.435088, 0.396491, 0.487719, 0.515789, 0.494737,
 0.487719, 0.473684, 0.533333, 0.649123, 0.705263, 0.705263, 0.701754, 0.68
7719, 0.698246]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0526316, 0.0561404, 0.122807, 0.242105, 0.666667, 0.78596
5, 0.845614, 0.85614, 0.863158, 0.863158, 0.880702, 0.877193, 0.891228, 0.8
80702, 0.877193]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0526316, 0.0526316, 0.0596491, 0.119298, 0.350877, 0.5929
82, 0.670175, 0.726316, 0.764912, 0.792982, 0.82807, 0.859649, 0.849123, 0.
863158, 0.852632]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0526316, 0.0526316, 0.0526316, 0.0526316, 0.126316, 0.326
316, 0.375439, 0.368421, 0.459649, 0.564912, 0.65614, 0.663158, 0.705263, 0
.684211, 0.712281]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.108772, 0.136842, 0.164912, 0.266667, 0.673684, 0.782456,
 0.85614, 0.870175, 0.877193, 0.873684, 0.880702, 0.877193, 0.887719, 0.891
228, 0.873684]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0385965, 0.0526316, 0.0526316, 0.0526316, 0.105263, 0.105
263, 0.105263, 0.105263, 0.105263, 0.105263, 0.368421, 0.473684, 0.526316, 
0.526316, 0.578947]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0, 0.00701754, 0.00701754, 0.0105263, 0.0491228, 0.052631
6, 0.0526316, 0.0561404, 0.161404, 0.431579, 0.561404, 0.677193, 0.698246, 
0.761404, 0.792982]
 BenchmarkResults :
Run length : [10, 16, 27, 44, 72, 118, 193, 316, 518, 848, 1389, 2276, 3728
, 6105, 10000]
Success rate : [0.0, 0.00350877, 0.0421053, 0.0280702, 0.0526316, 0.0526316
, 0.0526316, 0.0561404, 0.133333, 0.263158, 0.494737, 0.557895, 0.589474, 0
.635088, 0.705263]
```





## Success Rate vs. Function Evaluations

```julia
labels = collect(keys(setup))
idx = sortperm([b.success_rate[end] for b in results], rev = true)

plot_success_curves(
    [results[i].callcount for i in idx],
    [results[i].success_rate for i in idx],
    labels[idx];
    xlabel = "Function evaluations", xlim = (1, 1e7))
```

![](figures/pso_global_optimizers_6_1.png)



## Success Rate vs. Iterations

```julia
labels = collect(keys(setup))
idx = sortperm([b.success_rate[end] for b in results], rev = true)

plot_success_curves(
    [results[i].run_length for i in idx],
    [results[i].success_rate for i in idx],
    labels[idx];
    xlabel = "Iterations", xlim = (1, 1e5))
```

![](figures/pso_global_optimizers_7_1.png)



## Test all (wall-clock time to success)

```julia
tts_results = Dict{String, Vector{Float64}}()

for algo in labels
    algo in PSO_KEYS || continue
    tts_results[algo] = run_tts(algo)
end

for algo in labels
    algo in PSO_KEYS && continue
    tts_results[algo] = run_tts(algo)
end
```




## Success Rate vs. Wall-Clock Time

```julia
labels = collect(keys(setup))

all_finite      = filter(isfinite, vcat(values(tts_results)...))
t_lo            = minimum(all_finite) / 2
t_hi            = maximum(all_finite) * 2
time_thresholds = 10 .^ range(log10(t_lo), log10(t_hi), length = 50)

cdfs = Dict(algo => success_rate_cdf(tts_results[algo], time_thresholds) for algo in labels)
idx  = sortperm([cdfs[l][end] for l in labels], rev = true)

plot_success_curves(
    [time_thresholds for _ in idx],
    [cdfs[labels[i]] for i in idx],
    labels[idx];
    xlabel = "Wall time (s)")
```

![](figures/pso_global_optimizers_9_1.png)



## Success Rate per Function Heatmap

```julia
success_rate_per_function = reduce(hcat, b.success_rate_per_function for b in results)

idx = sortperm(mean(success_rate_per_function, dims = 1)[:], rev = false)
idxfunc = 1:length(test_functions)

n_func = length(idxfunc)
n_algo = length(idx)
data = success_rate_per_function[idxfunc, idx]

fig = Figure(; size = (1100, 550))
ax = Axis(fig[1, 1];
    xticks = (1:n_func, string.(test_functions)[idxfunc]),
    yticks = (1:n_algo, labels[idx]),
    xticklabelrotation = π / 4)
hm = heatmap!(ax, 1:n_func, 1:n_algo, data; colormap = :RdYlGn, colorrange = (0, 1))
Colorbar(fig[1, 2], hm; label = "Success rate")
fig
```

![](figures/pso_global_optimizers_10_1.png)



## Distance to Minimizer vs. Iterations

```julia
labels = collect(keys(setup))
idx = sortperm([b.distance_to_minimizer[end] for b in results], rev = false)

plot_success_curves(
    [results[i].run_length for i in idx],
    [results[i].distance_to_minimizer for i in idx],
    labels[idx];
    xlabel = "Iterations", ylabel = "Mean distance to minimum",
    xlim = (1, 1e5), ylim = (0, 5), size = (PLOT_WIDTH, 500))
```

![](figures/pso_global_optimizers_11_1.png)



## Relative Runtime

```julia
labels = collect(keys(setup))
ref = findfirst("NelderMead" .== labels)
runtimes = getfield.(results, :runtime)
runtimes = runtimes ./ runtimes[ref]

fig = Figure(; size = (PLOT_WIDTH, 500))
ax = Axis(fig[1, 1]; ylabel = "Run time relative to NM", yscale = log10,
    xticks = (1:length(labels), labels), xticklabelrotation = π / 4, spinewidth = STROKEWIDTH)
barplot!(ax, 1:length(labels), runtimes; color = :steelblue, strokewidth = 1.5)
fig
```

![](figures/pso_global_optimizers_12_1.png)
