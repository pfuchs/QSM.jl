"""
    ikd(
        f::AbstractArray{<:AbstractFloat, N ∈ (3,4)},
        mask::AbstractArray{Bool, 3};
        pad::NTuple{3, Integer} = (0, 0, 0),
        Dkernel::Symbol = :k,
        bdir::NTuple{3, Real} = (0, 0, 1),
        lambda::Real = 0.2,
        tol::Real = 1e-1,
        maxit::Integer = 100,
        verbose::Bool = false,
    ) -> typeof(similar(f))

Incomplete Spectrum deconvolution using CG [1]

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) local
    field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest

### Keywords
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing`:
    data fidelity weights
- `Wtv::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, 5)}} = nothing`:
    total variation weights
    - `M = 3`: same weights for all three gradient directions and all echoes
    - `M = 4 = N`: same weights for all three gradient directions, different weights for echoes
    - `M = 5, (size(Wtv)[4,5] = [1 or N, 3]`: different weights for each gradient direction
- `pad::NTuple{3, Integer} = (0, 0, 0)`: zero padding array
    - `< 0`: no padding
    - `≥ 0`: minimum padding to fast fft size
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `Dkernel::Symbol = :k`: dipole kernel method
- `lambda::Real = 1e-3`: regularization parameter
- `rho::Real = 100*lambda`: Lagrange multiplier penalty parameter
- `mu::Real = 1`: Lagrange multiplier penalty parameter (unused if `W = nothing`)
- `tol::Real = 1e-3`: stopping tolerance
- `maxit::Integer = 250`: maximum number of iterations
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: susceptibility map

### References
[1] Fuchs P, Shmueli K. Incomplete Spectrum Quantitative Susceptibility Mapping
"""
function ikd(
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real};
    pad::NTuple{3, Integer} = (0, 0, 0),
    Dkernel::Symbol = :k,
    bdir::NTuple{3, Real} = (0, 0, 1),
    lambda::Real = 0.2,
    tol::Real = 1e-1,
    maxit::Integer = 100,
    verbose::Bool = false,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _ikd!(
        tzero(f), f, mask, vsz, pad, Dkernel, bdir,lambda, tol, maxit, verbose
    )
end

function _ikd!(
    x::AbstractArray{<:AbstractFloat, N},
    f::AbstractArray{T, N},
    mask::AbstractArray{Bool, 3},
    vsz::NTuple{3, Real},
    pad::NTuple{3, Integer},
    Dkernel::Symbol,
    bdir::NTuple{3, Real},
    lambda::Real,
    tol::Real,
    maxit::Integer,
    verbose::Bool,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(x, f, (:x, :f))
    checkshape(axes(mask), axes(f)[1:3], (:mask, :f))

    checkopts(Dkernel, (:k, :kspace, :i, :ispace), :Dkernel)

    # convert scalars
    zeroT = zero(T)

    λ = convert(T, lambda)
    ϵ = convert(T, tol)

    # pad to fast fft size
    xp = padfastfft(@view(f[:,:,:,1]), pad, rfft=true)
    m = padfastfft(mask, pad, rfft=true)

    # initialize variables and fft
    sz0 = size(mask)
    sz  = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    x0 = similar(xp)    

    D  = Array{T}(undef, sz_)           # dipole kernel
    K  = Array{T}(undef, sz_)           # band-limit kernel

    X̂ = Array{complex(T)}(undef, sz_)   # in-place rfft var
    F̂ = Array{real(T)}(undef, sz)   # pre-computed rhs

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(xp)
    iP = inv(P)

    # get kernels
    D = _dipole_kernel!(D, X̂, xp, sz0, vsz, bdir, P, Dkernel, :rfft)
    @bfor K[I] = abs(D[I]) > λ

    for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        xp = padarray!(xp, @view(f[:, :, :, t]))

        # Perform direct deconvolution
        # b = D^-1 F M f
        @bfor xp[I] *= m[I]
        mul!(X̂, P, xp)
        @bfor begin
            b = D[I]*K[I]
            if iszero(b)
                X̂[I] = zeroT
            else
                X̂[I] = inv(b) * X̂[I]
            end
        end

        # Initalise rhs (normal equation)
        # b = A^H F̂^H
        mul!(F̂, iP, X̂)
        @bfor F̂[I] *= m[I]
        
        # cg
        A = LinearMap{complex(T)}(
            (Av, v) -> _A_ikd!(Av, v, K, m, P, iP, sz),
            length(m),
            ishermitian = true,
            ismutating = true
        )

        verbose && @printf("\n iter\tresidual\n")

        cg!(vec(xp), A, vec(F̂); abstol=ϵ, maxiter=maxit, verbose=verbose)
        xp = reshape(xp, sz)

        # ##################################################################
        # # Initalise CG iterations
        # ##################################################################
        # # r0 = A x0 - b; p0 = - r0; i = 0
        # r = similar(xp)
        # p = similar(xp)
        # c = similar(xp)

        # # r0 = A x0 - b
        # copyto!(r, xp)
        # mul!(r, P, r)
        # @bfor r[I] *= K[I]
        # mul!(r, iP, r)
        # @bfor r[I] muladd(m[I], r[I], -F̂[I])
        
        # # p0 = - r0
        # p .-= r
        
        # residual = norm(r)

        # if verbose
        #     @printf("\n iter\t  ||x-xprev||/||x||\n")
        # end

        # for i in 1:maxit
        #     x0, xp = xp, x0

        #     # Check for termination first
        #     if done(maxit, i, ϵ, residual)
        #         return nothing
        #     end
        #     ##################################################################
        #     # CG 
        #     ##################################################################
            
        #     # c = A * pk
        #     mul!(c, P, p)
        #     @bfor c[I] *= K[I]
        #     mul!(c, iP, c)
        #     @bfor c[I] *= m[I]

        #     pTAp = zeroT
        #     @bfor pTAp += p[I]*c[I]

        #     α = residual^2 / pTAp

        #     # Improve solution and residual
        #     xp .+= α .* p
        #     r .-= α .* c

        #     prev_residual = residual
        #     @batch threadlocal=zeros(T, 1)::Vector{T} for I in eachindex(r)
        #         a = r[I]
        #         threadlocal = muladd(a, a, threadlocal)
        #     end
        #     residual = sqrt.(sum(threadlocal::Vector{Vector{T}}))

        #     # u := r + βu (almost an axpy)
        #     β = residual^2 / prev_residual^2
        #     @bfor p[I] = r[I] + β * p[I]          

        #     if verbose
        #         @printf("%3d/%d\t    %.4e\n", i, maxit, residual)
        #     end

        # end

        unpadarray!(@view(x[:,:,:,t]), xp)
        
        if verbose
            println()
        end

    end

    return x
end

# @inline converged(tol::Real, residual::Real) = residual ≤ tol

# @inline done(maxiter::Int, iteration::Int, tol::Real, residual::Real) = iteration ≥ maxiter || converged(tol, residual)

function _A_ikd!(Av, v, K, m, P, iP, sz)
    Av = reshape(Av, sz)
    v = reshape(v, sz)
    v̂ = P*v
    # @bfor v[I] *= m[I]
    # mul!(Av, P, v)
    @bfor v̂[I] *= K[I]
    mul!(Av, iP, v̂)
    @bfor Av[I] *= m[I]
    return vec(Av)
end