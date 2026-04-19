include("bbob.jl")  # or wherever you saved the GPU BBOB file

using KernelAbstractions
using KernelAbstractions: CPU
import Enzyme: autodiff_deferred, Active, Reverse, Const

_dot(a, b) = sum(a .* b)
_norm(a) = sqrt(_dot(a, a))

@inline function _enzyme_grad(f, p, x)
    autodiff_deferred(Reverse, Const(u -> f(u, p)), Active, Active(x))[1][1]
end

@kernel function lbfgs_descent!(f, p, x0s, result, lb, ub, maxiters)
    i = @index(Global, Linear)
    x = clamp.(x0s[i], lb, ub)
    fx = f(x, p)
    g = _enzyme_grad(f, p, x)
    s = zero(x)
    y = zero(x)
    Hd = one(eltype(x))

    for iter in 1:maxiters
        _norm(g) < 1e-10 && break
        any(isnan, g) && break

        if iter > 1
            ys = _dot(y, s)
            if ys > 1e-12
                Hd = ys / _dot(y, y)
                rho = 1.0 / ys
                ah = rho * _dot(s, g)
                q = g - ah * y
                z = Hd * q
                b = rho * _dot(y, z)
                dir = -(z + s * (ah - b))
            else
                dir = -Hd * g
            end
        else
            dir = -g
        end

        al = one(eltype(x))
        gd = _dot(g, dir)
        gd >= 0 && break

        x_new = x
        for _ in 1:20
            x_trial = clamp.(x + al * dir, lb, ub)
            ft = f(x_trial, p)
            if isfinite(ft) && ft < fx + 1e-4 * al * gd
                x_new = x_trial
                break
            end
            al *= eltype(x)(0.5)
        end

        x_new == x && break

        g_new = _enzyme_grad(f, p, x_new)
        s = x_new - x
        y = g_new - g
        fx = f(x_new, p)
        x = x_new
        g = g_new
    end
    @inbounds result[i] = x
end

suite, names = gpu_bbob_suite(Val(3))
lb = SVector{3}(Float32(-5), Float32(-5), Float32(-5))
ub = SVector{3}(Float32(5), Float32(5), Float32(5))
kernel = lbfgs_descent!(CPU())

for (i, (bf, name)) in enumerate(zip(suite, names))
    obj = (x, p) -> bf(x)
    x0s = [SVector{3}(ntuple(_ -> Float32(-5 + rand() * 10), Val(3))) for _ in 1:256]
    result = similar(x0s)
    kernel(obj, nothing, x0s, result, lb, ub, 50; ndrange = 256)
    best = minimum(x -> obj(x, nothing), result)
    gap = abs(best - bf.f_opt)
    status = gap < 1e-4 ? "pass" : "fail"
    println("$(rpad(name, 30)) gap=$(round(gap, sigdigits=3))  $(status)")
end