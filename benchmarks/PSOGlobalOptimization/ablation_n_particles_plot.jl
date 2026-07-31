using Random
Random.seed!(42)

using BlackBoxOptimizationBenchmarking, Statistics, LinearAlgebra, ForwardDiff
using Optimization, StaticArrays, CUDA, KernelAbstractions, ParallelParticleSwarms, CairoMakie
import BlackBoxOptimizationBenchmarking: BBOBFunction
const BBOB = BlackBoxOptimizationBenchmarking

const D = 10
const Δf = 1e-6
const NTRIALS = 10
const LOCAL_MAXITERS = 50
const RUN_LENGTH = round.(Int, 10 .^ LinRange(1, 3.3, 8))   # ~10 … 2000
const N_PARTICLES = [64, 128, 256, 512, 1024, 5000]
const BACKEND = CUDABackend()
const OUTDIR = @__DIR__

const f = only(filter(ff -> nameof(ff.f) == :f8, BBOB.bbob_suite(Val(D))))

# --- problem setup (matches pso_global_optimizers.jmd) ---

_to_f64(x::Real) = Float64(x)
_to_f64(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))
_to_f64(x) = Float64(x[])
_value(x::Real) = x
_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_penalty(x) = eltype(x) <: ForwardDiff.Dual ? zero(first(x)) + 1.0f10 : 1.0f10

function pso_objective(f::BBOBFunction, x)
    any(xi -> !isfinite(_value(xi)) || abs(_value(xi)) > 15, x) && return _penalty(x)
    y = f(x)
    y isa ForwardDiff.Dual ? y : Float32(y)
end

function make_prob()
    optf = OptimizationFunction{false}((x, p) -> pso_objective(f, x), Optimization.SciMLBase.NoAD())
    lb = SVector{D, Float32}(ntuple(_ -> -5.0f0, Val(D)))
    ub = SVector{D, Float32}(ntuple(_ -> 5.0f0, Val(D)))
    x0 = SVector{D, Float32}(ntuple(_ -> -5.0f0 + rand(Float32) * 10.0f0, Val(D)))
    OptimizationProblem{false}(optf, x0, nothing; lb, ub)
end

hybrid(n) = HybridPSO(;
    pso = ParallelSyncPSOKernel(n; backend = BACKEND),
    backend = BACKEND,
)

function solve_hybrid(n, maxiters)
    solve(make_prob(), hybrid(n);
        maxiters,
        local_maxiters = LOCAL_MAXITERS,
        abstol = 1.0f-8,
        reltol = 1.0f-8,
    )
end

# --- ablation ---

function run_ablation()
    curves = Dict{Int, NamedTuple}()
    for n in N_PARTICLES
        sr = Float64[]
        gaps = Float64[]
        for rl in RUN_LENGTH
            hits = 0
            gap_sum = 0.0
            for _ in 1:NTRIALS
                sol = solve_hybrid(n, rl)
                gap = abs(_to_f64(sol.objective) - f.f_opt)
                hits += gap < Δf
                gap_sum += gap
            end
            push!(sr, hits / NTRIALS)
            push!(gaps, gap_sum / NTRIALS)
        end
        curves[n] = (;
            sr,
            gaps,
            callcount = Float64.(RUN_LENGTH) .* n,  # iters × n (PSO stage)
        )
    end
    return curves
end

function plot_ablation(curves)
    markers = [:circle, :rect, :utriangle, :diamond, :dtriangle, :pentagon]
    fig = Figure(size = (1250, 820), fontsize = 15)
    Label(fig[0, 1:2],
        "HybridPSO n_particles ablation — f8 Rosenbrock (D=$D), SyncPSO global + local refine";
        fontsize = 17, font = :bold, tellwidth = false)

    ax1 = Axis(fig[1, 1]; xscale = log10, xlabel = "Iterations (PSO stage)",
        ylabel = "Success rate", title = "Success vs iterations",
        limits = (nothing, nothing, -0.05, 1.05))
    ax2 = Axis(fig[1, 2]; xscale = log10, xlabel = "Function evaluations (iters × n)",
        ylabel = "Success rate", title = "Success vs fevals",
        limits = (nothing, nothing, -0.05, 1.05))
    ax3 = Axis(fig[2, 1]; xscale = log10, yscale = log10, xlabel = "Iterations (PSO stage)",
        ylabel = "Mean |f − f*|", title = "Gap vs iterations")
    ax4 = Axis(fig[2, 2]; xscale = log10, yscale = log10,
        xlabel = "Function evaluations (iters × n)",
        ylabel = "Mean |f − f*|", title = "Gap vs fevals")

    for (j, n) in enumerate(N_PARTICLES)
        c = curves[n]
        kw = (; label = "n = $n", linewidth = 2.5, markersize = 9,
              marker = markers[mod1(j, length(markers))])
        scatterlines!(ax1, Float64.(RUN_LENGTH), c.sr; kw...)
        scatterlines!(ax2, c.callcount, c.sr; kw...)
        scatterlines!(ax3, Float64.(RUN_LENGTH), max.(c.gaps, 1e-16); kw...)
        scatterlines!(ax4, c.callcount, max.(c.gaps, 1e-16); kw...)
    end

    Legend(fig[1:2, 3], ax1; framevisible = false)
    return fig
end

# --- main ---

solve_hybrid(32, 3)  # warmup
curves = run_ablation()
fig = plot_ablation(curves)
save(joinpath(OUTDIR, "ablation_f8_hybrid_n_particles.png"), fig)
