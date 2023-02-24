"""
    ikd(
        f::AbstractArray{<:AbstractFloat, N ∈ (3,4)},
        mask::AbstractArray{Bool, 3},
        vsz::NTuple{3, Real};
        pad::NTuple{3, Integer} = (0, 0, 0),
        Dkernel::Symbol = :k,
        bdir::NTuple{3, Real} = (0, 0, 1),
        delta::Real = 0.2,
        tol::Real = 1e-1,
        maxit::Integer = 100,
        verbose::Bool = false,
    ) -> typeof(similar(f))

Incomplete Spectrum deconvolution using CG [1]

### Arguments
- `f::AbstractArray{<:AbstractFloat, N ∈ (3, 4)}`: unwrapped (multi-echo) local
    field/phase
- `mask::AbstractArray{Bool, 3}`: binary mask of region of interest
- `vsz::NTuple{3, Real}`: voxel size

### Keywords
- `pad::NTuple{3, Integer} = (0, 0, 0)`: zero padding array
    - `< 0`: no padding
    - `≥ 0`: minimum padding to fast fft size
- `bdir::NTuple{3, Real} = (0, 0, 1)`: unit vector of B field direction
- `Dkernel::Symbol = :k`: dipole kernel method
- `delta::Real = 0.2`: threshold for ill-conditioned k-space region
- `rho::Real = 100*delta`: Lagrange multiplier penalty parameter
- `mu::Real = 1`: Lagrange multiplier penalty parameter (unused if `W = nothing`)
- `tol::Real = 1e-1`: stopping tolerance
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
    delta::Real = 0.25,
    tol::Real = 1e-1,
    maxit::Integer = 30,
    verbose::Bool = false,
) where {T, N}
    N ∈ (3, 4) || throw(ArgumentError("arrays must be 3d or 4d, got $(N)d"))
    return _ikd!(
        tzero(f), f, mask, vsz, pad, Dkernel, bdir, delta, tol, maxit, verbose
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
    delta::Real,
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

    δ = convert(T, delta)
    ϵ = convert(T, tol)

    # pad to fast fft size
    xp = padfastfft(@view(f[:,:,:,1]), pad, rfft=true)
    m = padfastfft(mask, pad, rfft=true)

    # initialize variables and fft
    sz0 = size(mask)
    sz  = size(m)
    sz_ = (sz[1]>>1 + 1, sz[2], sz[3])

    D  = Array{T, 3}(undef, sz_)            # dipole kernel
    K  = Array{T}(undef, sz_)               # band-limit
    F̂ = Array{complex(T), 3}(undef, sz_)    # in-place rfft var

    b = Array{real(T), 3}(undef, sz)    # pre-computed rhs

    FFTW.set_num_threads(FFTW_NTHREADS[])
    P = plan_rfft(xp)
    iP = inv(P)

    x0 = similar(xp)

    # get dipole kernel
    D = _dipole_kernel!(D, F̂, xp, sz0, vsz, bdir, P, Dkernel, :rfft)

    # band-limit
    @bfor K[I] = abs(D[I]) > δ

    # inverse k-space kernel
    # @bfor D[I] = inv(D[I])
    # iD = D

    for t in axes(f, 4)
        if verbose && size(f, 4) > 1
            @printf("Echo: %d/%d\n", t, size(f, 4))
        end

        xp = padarray!(xp, @view(f[:, :, :, t]))

        # Perform direct deconvolution
        # b = D^-1 F M f
        @bfor xp[I] *= m[I]
        F̂ = mul!(F̂, P, xp)

        @bfor begin
            b = D[I]*K[I]
            if iszero(b)
                F̂[I] = zeroT
            else
                F̂[I] = inv(b) * F̂[I]
            end
        end

        # Initalise rhs (normal equation)
        # b = A^H F̂^H
        b = mul!(b, iP, F̂)
        @bfor b[I] *= m[I]
        
        # cg
        A = LinearMap{complex(T)}(
            (Av, v) -> _A_ikd!(Av, v, K, m, P, iP, sz),
            length(m),
            ishermitian = true,
            ismutating = true
        )

        verbose && @printf("\n iter\tresidual\n")

        cg!(vec(xp), A, vec(b); abstol=ϵ, maxiter=maxit, verbose=verbose)
        xp = reshape(xp, sz)

        unpadarray!(@view(x[:,:,:,t]), xp)
        
        if verbose
            println()
        end

    end

    return x
end

function _A_ikd!(Av, v, K, m, P, iP, sz)
    Av = reshape(Av, sz)
    v = reshape(v, sz)
    v̂ = P*v
    @bfor v̂[I] *= K[I]
    mul!(Av, iP, v̂)
    @bfor Av[I] *= m[I]
    return vec(Av)
end