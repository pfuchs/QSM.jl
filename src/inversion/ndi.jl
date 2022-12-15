"""
    ndi(
        f::AbstractArray{T, N}
        vsz::NTuple{3, Real};
        alpha::Real = 0.2,
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
- `alpha::Real = 1e-3`: regularization parameter
- `rho::Real = 100*alpha`: Lagrange multiplier penalty parameter
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
    f::AbstractArray{T, N},
    vsz::NTuple{3, Real};
    alpha::Real = 1e-5,
    tol::Real = 1e-2,
    maxit::Integer = 100,
    W::Union{Nothing, AbstractArray{<:AbstractFloat}} = nothing,
    tau::Real = 2.0,
    pad::NTuple{3, Integer} = (0, 0, 0),
    Dkernel::Symbol = :k,
    bdir::NTuple{3, Real} = (0, 0, 1),
    verbose::Bool = false
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _ndi!(
        tzero(f), f, vsz, alpha, tol, maxit, W, tau, pad, Dkernel, bdir, verbose
    )
end

function _ndi!(
    x::AbstractArray{<:AbstractFloat, N},
    f::AbstractArray{T, N},
    vsz::NTuple{3, Real},
    alpha::Real,
    tol::Real,
    maxit::Integer,
    W::Union{Nothing, AbstractArray{<:AbstractFloat}},
    tau::Real,
    pad::NTuple{3, Integer},
    Dkernel::Symbol,
    bdir::NTuple{3, Real},
    verbose::Bool,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))

    checkshape(x, f, (:x, :f))

    checkopts(Dkernel, (:k, :kspace, :i, :ispace), :Dkernel)

    if W !== nothing
        checkshape(Bool, axes(W), axes(f)[1:3]) ||
        checkshape(W, f, (:W, :f))
    end

    # convert scalars
    zeroT = zero(T)

    α = convert(T, alpha)
    τ = convert(T, tau)
    ϵ = convert(T, tol)

    # pad to fast fft size
    xp = padfastfft(@view(f[:,:,:,1]), pad, rfft=true)


    # initialize variables and fft
    sz0 = size(@view(f[:,:,:,1]))
    sz  = size(xp)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    fp = similar(xp)
    x0 = similar(xp)
    x̂ = similar(xp)

    if W !== nothing
        Wp = similar(xp)
    end

    D  = Array{T}(undef, sz_)           # dipole kernel
    K  = Array{T}(undef, sz_)           # band-limit kernel

    X̂ = Array{complex(T)}(undef, sz_)   # in-place rfft var
    F̂ = Array{real(T)}(undef, sz)   # pre-computed rhs

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(xp)
    iP = inv(P)

    # get kernels
    D = _dipole_kernel!(D, X̂, xp, sz, vsz, bdir, P, Dkernel, :rfft)
    DT = conj(D)

    for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        fp = padarray!(fp, @view(f[:, :, :, t]))
        xp = copyto!(xp, fp)
        # x0 = zeros(T, size(fp))

        if W !== nothing
            Wp = padarray!(Wp, @view(W[:,:,:,min(t, end)]))
        end

        verbose &&   @printf("\n iter\t  ||x-xprev||/||x||\n")
        
        for i in 1:maxit
            x0, xp = xp, x0

            x̂ = copyto!(x̂, xp)
            susc2field!(x̂, X̂, D, P, iP)
            
            @bfor x̂[I] = sin( x̂[I] - fp[I] )
            if W !== nothing
                @bfor x̂[I] *= Wp[I] 
            end

            susc2field!(x̂, X̂, DT, P, iP)

            @bfor x0[I] = xp[I] - tau * x̂[I] - tau * alpha * x0[I]
            
            # mul!(X̂, P, x0)
            # @bfor X̂[I] *= D[I]
            # mul!(x0, iP, X̂)

            # @bfor x0[I] -= fp[I]
            # @bfor x0[I] = sin(x0[I])

            # mul!(X̂, P, x0)
            # @bfor X̂[I] *= DT[I]
            # mul!(x0, iP, X̂)

            # @bfor x0[I] *= τ
            # @bfor x0[I] += x0[I] * (1 - τ * α)


            # x0 = xp - τ * (iP * (DT .* (P * ( (iP * (D .* (P * x0 ))) - fp)))) - τ * α * x0

            @batch threadlocal=zeros(T, 2)::Vector{T} for I in eachindex(xp)
                a, b = xp[I], x0[I]
                threadlocal[1] = muladd(a-b, a-b, threadlocal[1])
                threadlocal[2] = muladd(a, a, threadlocal[2])
            end
            ndx, nx = sqrt.(sum(threadlocal::Vector{Vector{T}}))

            # e1, e2 = freq_energy(xp, m1, m2, P)

            verbose && @printf("%3d\t   %.4f\n", i, ndx/nx)
            
            # if (e1 > e2) && (i > 3)
                # verbose && @printf("Early stopping reached.\n")
                # break
            # end
            if ndx < ϵ*nx || i == maxit
                break
            end
        end

        unpadarray!(@view(x[:,:,:,t]), x0)

        if verbose
            println()
        end

    end

    return x
end

function susc2field!(v, X̂, D, P, iP)
    mul!(X̂, P, v)
    @bfor X̂[I] *= D[I]
    mul!(v, iP, X̂)
    return v
end

function freq_energy( x, m1, m2, P)
    fx = P*x
    
    n1 = sum(m1)
    n2 = sum(m2)

    e1 = sum( fx*m1 ) / n1
    e2 = sum( fx*m2 ) / n2
    return e1, e2
end