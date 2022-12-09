"""
    ndi(
        f::AbstractArray{T, N}
        vsz::NTuple{3, Real};
        lambda::Real = 0.2,
        tol::Real = 1e-1,
        maxit::Integer = 100,
        W::AbstractArray{T, M ∈ (3, N)},
        tau::Real = 2.0,
        pad::NTuple{3, Integer} = (0, 0, 0),
        Dkernel::Symbol = :k,
        bdir::NTuple{3, Real} = (0, 0, 1),
        verbose::Bool = false,
    ) -> typeof(similar(f))

Incomplete Spectrum deconvolution using admm [1]

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) local
    field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest

### Keywords
- `lambda::Real = 1e-3`: regularization parameter
- `rho::Real = 100*lambda`: Lagrange multiplier penalty parameter
- `mu::Real = 1`: Lagrange multiplier penalty parameter (unused if `W = nothing`)
- `tol::Real = 1e-3`: stopping tolerance
- `maxit::Integer = 250`: maximum number of iterations
- `W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing`:
    data fidelity weights
- `pad::NTuple{3, Integer} = (0, 0, 0)`: zero padding array
    - `< 0`: no padding
    - `≥ 0`: minimum padding to fast fft size
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `Dkernel::Symbol = :k`: dipole kernel method
- `verbose::Bool = false`: print convergence information

### Returns
- `typeof(similar(f))`: susceptibility map

### References
[1] Polak D, et al. NMR Biomed 2020
"""
function ndi(
    f::AbstractArray{T, N}
    vsz::NTuple{3, Real};
    lambda::Real = 0.2,
    tol::Real = 1e-1,
    maxit::Integer = 100,
    W::Union{Nothing, AbstractArray{<:AbstractFloat, M ∈ (3, N)}} = nothing,
    tau::Real = 2.0,
    pad::NTuple{3, Integer} = (0, 0, 0),
    Dkernel::Symbol = :k,
    bdir::NTuple{3, Real} = (0, 0, 1),
    verbose::Bool = false,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _ikd!(
        tzero(f), f, vsz, lambda, tol, maxit, W, tau, pad, Dkernel, bdir, verbose
    )
end

function _ndi!(
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


        verbose &&   @printf("\n iter\t  ||x-xprev||/||x||\n")
        for i in 1:maxit
            x0, xp = xp, x0

            @bfor xp[I] -= tau*alpha*x0[I]

            X̂ = P * x0
            @bfor X̂[I] *= D[I]
            x0 = iP * X̂

            x0 = sin( x0 - x )
            @bfor x0 = W[I] * sin( x0[I] - x[I] )
            X̂ = P * x0
            @bfor X̂[I] *= conj(D[I])
            x0 = iP * X̂
            @bfor x0[I] += xp[I] 
            
            @batch threadlocal=zeros(T, 2)::Vector{T} for I in eachindex(xp)
                a, b = xp[I], x0[I]
                threadlocal[1] = muladd(a-b, a-b, threadlocal[1])
                threadlocal[2] = muladd(a, a, threadlocal[2])
            end
            ndx, nx = sqrt.(sum(threadlocal::Vector{Vector{T}}))

            verbose && @printf("%3d\t   %.4f\n", i, ndx/nx)
            
            if ndx < ϵ*nx || i == maxit
                break
            end

            unpadarray!(@view(x[:,:,:,t]), xp)
        end
    end

    return x
end