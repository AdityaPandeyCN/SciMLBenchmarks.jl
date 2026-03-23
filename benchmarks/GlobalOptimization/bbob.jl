# GPU-compatible BBOB (Black-Box Optimization Benchmarking) test suite
using StaticArrays, LinearAlgebra, Random

struct GPUBBOBFunction{F, N, M}
    f::F                             # one of bbob_sphere, bbob_rastrigin, ...
    x_opt::SVector{N, Float32}       # shifted optimum location
    f_opt::Float32                    # shifted optimum value
    Q::SMatrix{N, N, Float32, M}     # first rotation matrix
    R::SMatrix{N, N, Float32, M}     # second rotation matrix
end

(func::GPUBBOBFunction)(x) = func.f(x, func.x_opt, func.f_opt, func.Q, func.R)
Base.show(io::IO, f::GPUBBOBFunction) = print(io, string(f.f))

function make_rotation(::Val{N}, seed::Int) where N
    rng = MersenneTwister(seed)
    A = randn(rng, Float32, N, N)
    SMatrix{N, N, Float32}(Matrix(qr(A).Q))
end

function make_x_opt(::Val{N}, seed::Int) where N
    rng = MersenneTwister(seed)
    SVector{N, Float32}(rand(rng, Float32, N) .* 10f0 .- 5f0)
end

function make_f_opt(seed::Int)
    rng = MersenneTwister(seed)
    Float32(clamp(round(randn(rng) * 100, digits = 2), -1000, 1000))
end

## Oscillation transform (element-wise)
@inline function t_osz(xi::T) where T
    xhat = xi != zero(T) ? log(abs(xi)) : zero(T)
    c1 = xi > zero(T) ? T(10) : T(5.5)
    c2 = xi > zero(T) ? T(7.9) : T(3.1)
    sign(xi) * exp(xhat + T(0.049) * (sin(c1 * xhat) + sin(c2 * xhat)))
end

@inline t_osz_vec(x::SVector) = map(t_osz, x)

## Asymmetric transform
@inline function t_asy(x::SVector{N, T}, β) where {N, T}
    SVector{N, T}(ntuple(Val(N)) do i
        x[i] > zero(T) ? x[i]^(one(T) + T(β) * T(i - 1) / T(N - 1) * sqrt(x[i])) : x[i]
    end)
end

## Diagonal scaling: Λ^α with exponents spread across dimensions
@inline function lambda_diag(::Val{N}, α::T) where {N, T}
    SVector{N, T}(ntuple(i -> α^(T(0.5) * T(i - 1) / T(N - 1)), Val(N)))
end

@inline function lambda_mul(::Val{N}, α::T, x::SVector{N, T}) where {N, T}
    lambda_diag(Val(N), α) .* x
end

## Boundary penalty: penalises components outside [-5, 5]
@inline function f_pen(x::SVector{N, T}) where {N, T}
    sum(max.(zero(T), abs.(x) .- T(5)) .^ 2)
end

## Precomputed weight vector for ellipsoidal-type functions
@inline function ellip_weights(::Val{N}, ::Type{T}, exponent::T) where {N, T}
    SVector{N, T}(ntuple(i -> T(10)^(exponent * T(i - 1) / T(N - 1)), Val(N)))
end

## F1: Sphere — simplest unimodal, sanity check
function bbob_sphere(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = x .- x_opt
    sum(z .^ 2) + f_opt
end

## F2: Ellipsoidal — ill-conditioned unimodal with oscillation
function bbob_ellipsoidal(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = t_osz_vec(x .- x_opt)
    w = ellip_weights(Val(N), T, T(6))
    sum(w .* z .^ 2) + f_opt
end

## F3: Rastrigin — multimodal with regular grid of local minima
function bbob_rastrigin(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = lambda_mul(Val(N), T(10), t_asy(t_osz_vec(x .- x_opt), T(0.2)))
    T(10) * (T(N) - sum(cos.(T(2) * T(π) .* z))) + sum(z .^ 2) + f_opt
end

## F4: Büche-Rastrigin — Rastrigin with coordinate-dependent scaling + penalty
function bbob_buche_rastrigin(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = t_osz_vec(x .- x_opt)
    s = SVector{N, T}(ntuple(i -> isodd(i) ? T(10) * T(10)^(T(0.5) * T(i - 1) / T(N - 1)) :
                                              T(10)^(T(0.5) * T(i - 1) / T(N - 1)), Val(N)))
    z = s .* z
    T(10) * (T(N) - sum(cos.(T(2) * T(π) .* z))) + sum(z .^ 2) + T(100) * f_pen(x) + f_opt
end

## F5: Linear Slope — only BBOB function with a non-quadratic global structure
function bbob_linear_slope(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    s = SVector{N, T}(ntuple(i -> sign(x_opt[i]) * T(10)^(T(i - 1) / T(N - 1)), Val(N)))
    z = SVector{N, T}(ntuple(i -> x_opt[i] * x[i] < T(25) ? x[i] : T(x_opt[i]), Val(N)))
    sum(T(5) .* abs.(s) .- s .* z) + f_opt
end

## F6: Attractive Sector — unimodal, different curvature in different sectors
function bbob_attractive_sector(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = Q * lambda_mul(Val(N), T(10), R * (x .- x_opt))
    z = SVector{N, T}(ntuple(i -> x_opt[i] * z[i] > zero(T) ? T(100) * z[i] : z[i], Val(N)))
    t_osz(sum(z .^ 2))^T(0.9) + f_opt
end

## F7: Step Ellipsoidal — piecewise-flat plateaus make gradient methods struggle
function bbob_step_ellipsoidal(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = lambda_mul(Val(N), T(10), R * (x .- x_opt))
    zhat_1 = z[1]
    z = SVector{N, T}(ntuple(i -> z[i] > T(0.5) ?
        floor(T(0.5) + z[i]) :
        floor(T(0.5) + T(10) * z[i]) / T(10), Val(N)))
    z = Q * z
    w = ellip_weights(Val(N), T, T(2))
    T(0.1) * max(abs(zhat_1) / T(1e4), sum(w .* z .^ 2)) + f_pen(x) + f_opt
end

## F8: Rosenbrock — classic banana-shaped valley
function bbob_rosenbrock(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = max(one(T), T(sqrt(N)) / T(8)) .* (x .- x_opt) .+ one(T)
    v = SVector{N-1, T}(ntuple(i -> T(100) * (z[i]^2 - z[i + 1])^2 + (z[i] - one(T))^2, Val(N - 1)))
    sum(v) + f_opt
end

## F9: Rosenbrock Rotated — rotated version to remove axis alignment
function bbob_rosenbrock_rotated(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = max(one(T), T(sqrt(N)) / T(8)) .* (R * (x .- x_opt)) .+ one(T)
    v = SVector{N-1, T}(ntuple(i -> T(100) * (z[i]^2 - z[i + 1])^2 + (z[i] - one(T))^2, Val(N - 1)))
    sum(v) + f_opt
end

## F10: Ellipsoidal 2 — rotated ill-conditioned ellipsoid
function bbob_ellipsoidal2(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = t_osz_vec(R * (x .- x_opt))
    w = ellip_weights(Val(N), T, T(2))
    sum(w .* z .^ 2) + f_opt
end

## F11: Discus — one direction has 10^6 higher curvature
function bbob_discus(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = t_osz_vec(R * (x .- x_opt))
    T(1e6) * z[1]^2 + sum(z .^ 2) - z[1]^2 + f_opt
end

## F12: Bent Cigar — one direction has 10^6 lower curvature
function bbob_bent_cigar(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = R * t_asy(R * (x .- x_opt), T(0.5))
    z[1]^2 + T(1e6) * (sum(z .^ 2) - z[1]^2) + f_opt
end

## F13: Sharp Ridge — narrow ridge, tests directional sensitivity
function bbob_sharp_ridge(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = Q * lambda_mul(Val(N), T(10), R * (x .- x_opt))
    z[1]^2 + T(100) * sqrt(sum(z .^ 2) - z[1]^2) + f_opt
end

## F14: Different Powers — each coordinate has a different exponent
function bbob_different_powers(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = R * (x .- x_opt)
    pw = SVector{N, T}(ntuple(i -> abs(z[i])^(T(2) + T(4) * T(i - 1) / T(N - 1)), Val(N)))
    sqrt(sum(pw)) + f_opt
end

## F15: Rastrigin 2 — fully rotated Rastrigin, harder than F3
function bbob_rastrigin2(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = R * lambda_mul(Val(N), T(10), Q * t_asy(t_osz_vec(R * (x .- x_opt)), T(0.2)))
    T(10) * (T(N) - sum(cos.(T(2) * T(π) .* z))) + sum(z .^ 2) + f_opt
end

## F16: Weierstrass — fractal-like landscape with nested cosine sums
function bbob_weierstrass(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = R * lambda_mul(Val(N), T(1 / 100), Q * t_osz_vec(R * (x .- x_opt)))
    f0 = zero(T)
    for k in 0:11
        f0 += T(1) / T(2)^k * cos(T(2) * T(π) * T(3)^k * T(0.5))
    end
    s = zero(T)
    for j in 1:N
        for k in 0:11
            s += T(1) / T(2)^k * cos(T(2) * T(π) * T(3)^k * (z[j] + T(0.5)))
        end
    end
    T(10) * (T(1) / T(N) * s - f0)^3 + T(10) / T(N) * f_pen(x) + f_opt
end

## F17: Schaffers F7 — rugged multimodal with sqrt + sin structure
function bbob_schaffers_f7(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = lambda_mul(Val(N), T(10), Q * t_asy(R * (x .- x_opt), T(0.5)))
    s = SVector{N - 1, T}(ntuple(i -> sqrt(z[i]^2 + z[i + 1]^2), Val(N - 1)))
    v = SVector{N - 1, T}(ntuple(i -> sqrt(s[i]) * (T(1) + sin(T(50) * s[i]^T(0.2))^2), Val(N - 1)))
    (T(1) / T(N - 1) * sum(v))^2 + T(10) * f_pen(x) + f_opt
end

## F18: Schaffers F7 Ill-Conditioned — F17 with 1000x scaling spread
function bbob_schaffers_f7_ill(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = lambda_mul(Val(N), T(1000), Q * t_asy(R * (x .- x_opt), T(0.5)))
    s = SVector{N - 1, T}(ntuple(i -> sqrt(z[i]^2 + z[i + 1]^2), Val(N - 1)))
    v = SVector{N - 1, T}(ntuple(i -> sqrt(s[i]) * (T(1) + sin(T(50) * s[i]^T(0.2))^2), Val(N - 1)))
    (T(1) / T(N - 1) * sum(v))^2 + T(10) * f_pen(x) + f_opt
end

## F19: Griewank-Rosenbrock — composition of Rosenbrock and Griewank
function bbob_griewank_rosenbrock(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = max(one(T), T(sqrt(N)) / T(8)) .* (R * (x .- x_opt)) .+ one(T)
    s = SVector{N - 1, T}(ntuple(i -> T(100) * (z[i]^2 - z[i + 1])^2 + (z[i] - one(T))^2, Val(N - 1)))
    v = SVector{N - 1, T}(ntuple(i -> s[i] / T(4000) - cos(s[i]), Val(N - 1)))
    T(10) / T(N - 1) * sum(v) + T(10) + f_opt
end

## F20: Schwefel — highly multimodal with deep non-global minima
function bbob_schwefel(x::SVector{N, T}, x_opt, f_opt, Q, R) where {N, T}
    z = T(100) .* (x .- T(2) .* abs.(x_opt))
    v = SVector{N, T}(ntuple(i -> z[i] * sin(sqrt(abs(z[i]))), Val(N)))
    -T(1) / (T(100) * T(N)) * sum(v) +
        T(4.189828872724339) + T(100) * f_pen(z ./ T(100)) + f_opt
end

const BBOB_FUNCTIONS = [
    bbob_sphere, bbob_ellipsoidal, bbob_rastrigin, bbob_buche_rastrigin, bbob_linear_slope,
    bbob_attractive_sector, bbob_step_ellipsoidal, bbob_rosenbrock, bbob_rosenbrock_rotated,
    bbob_ellipsoidal2, bbob_discus, bbob_bent_cigar, bbob_sharp_ridge, bbob_different_powers,
    bbob_rastrigin2, bbob_weierstrass, bbob_schaffers_f7, bbob_schaffers_f7_ill,
    bbob_griewank_rosenbrock, bbob_schwefel]

const BBOB_NAMES = [
    "F1  Sphere", "F2  Ellipsoidal", "F3  Rastrigin", "F4  Buche-Rastrigin",
    "F5  Linear Slope", "F6  Attractive Sector", "F7  Step Ellipsoidal",
    "F8  Rosenbrock", "F9  Rosenbrock Rotated", "F10 Ellipsoidal 2",
    "F11 Discus", "F12 Bent Cigar", "F13 Sharp Ridge", "F14 Different Powers",
    "F15 Rastrigin 2", "F16 Weierstrass", "F17 Schaffers F7",
    "F18 Schaffers F7 Ill-Cond", "F19 Griewank-Rosenbrock", "F20 Schwefel"]

"""
    gpu_bbob_suite(Val(N); seed=42) -> (suite, names)

Build the full 20-function BBOB suite for dimension `N`.
Each function gets deterministic random rotations and optima from `seed`.
"""
function gpu_bbob_suite(::Val{N}; seed = 42) where N
    suite = GPUBBOBFunction[]
    for (i, f) in enumerate(BBOB_FUNCTIONS)
        x_opt = make_x_opt(Val(N), seed + i)
        f_opt = make_f_opt(seed + 100 + i)
        Qmat  = make_rotation(Val(N), seed + 200 + i)
        Rmat  = make_rotation(Val(N), seed + 300 + i)
        push!(suite, GPUBBOBFunction(f, x_opt, f_opt, Qmat, Rmat))
    end
    suite, BBOB_NAMES
end