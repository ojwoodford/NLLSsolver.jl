using StaticArrays, Static

# The block layout of a Hessian partitioned into retained (c) and eliminable (p) variables:
#     H = [H_cc  H_cp]
#         [H_pc  H_pp]
# H_cc is dense, H_pp block diagonal, and H_cp block sparse with one block per (retained, eliminable)
# pair of variables that share a cost. Blocks are numbered retained first (1:ncblocks), then eliminable,
# grouped by size, and the unknowns are ordered the same way.
struct SchurStructure{G}
    nc::Int                                 # Dimension of the retained variables
    ncblocks::Int                           # Number of retained blocks
    blocksizes::Vector{Int}                 # One per block
    boffsets::Vector{Int}                   # Start index of each block in the unknowns
    ppoffsets::Vector{Int}                  # One per eliminable block: start index of its block in pp
    cpptr::Vector{Int}                      # One per eliminable block + 1: its range of coupling entries
    cpblock::Vector{Int}                    # One per coupling entry: the retained block index
    cpoffsets::Vector{Int}                  # One per coupling entry: start index of its block in cp
    groups::NTuple{G, UnitRange{Int}}       # Eliminable blocks (1-based, within the eliminable ones) of each size
end

# A coupling block between a retained block of size n_c and an eliminable block of size P is stored row
# major (n_c rows of P contiguous values); each diagonal block of H_pp is stored column major (P x P).
struct SchurBlockMatrix{T, G}
    cc::Matrix{T}
    pp::Vector{T}
    cp::Vector{T}
    structure::SchurStructure{G}
end

copy_structure(A::SchurBlockMatrix{T, G}) where {T, G} = SchurBlockMatrix{T, G}(zeros(T, size(A.cc)), zeros(T, length(A.pp)), zeros(T, length(A.cp)), A.structure)
zero!(A::SchurBlockMatrix) = (fill!(A.cc, 0); fill!(A.pp, 0); fill!(A.cp, 0); A)
add!(dest::SchurBlockMatrix, src::SchurBlockMatrix) = (dest.cc .+= src.cc; dest.pp .+= src.pp; dest.cp .+= src.cp; nothing)
Base.size(A::SchurBlockMatrix) = (n = schurdim(A); (n, n))
Base.size(A::SchurBlockMatrix, dim) = dim <= 2 ? size(A)[dim] : 1
Base.eltype(::SchurBlockMatrix{T}) where T = T
function schurdim(A::SchurBlockMatrix)
    s = A.structure
    return s.boffsets[end] + s.blocksizes[end] - 1
end
Base.axes(A::SchurBlockMatrix, dim) = Base.OneTo(size(A, dim))

# Call f(index, value) for each diagonal element of A, in the order of the unknowns, storing its result
@inline function mapdiag!(f::F, A::SchurBlockMatrix) where F
    s = A.structure
    cc = A.cc
    @inbounds for i in 1:s.nc
        cc[i,i] = f(i, cc[i,i])
    end
    pp = A.pp
    @inbounds for k in eachindex(s.ppoffsets)
        blk = s.ncblocks + k
        P = s.blocksizes[blk]
        o = s.ppoffsets[k]
        xo = s.boffsets[blk] - 1
        for d in 1:P
            j = o + (d - 1) * (P + 1)
            pp[j] = f(xo + d, pp[j])
        end
    end
    return A
end

uniformscaling!(A::SchurBlockMatrix, k) = mapdiag!((i, v) -> v + k, A)

# x' A x
function fast_bAb(A::SchurBlockMatrix, x::AbstractVector)
    s = A.structure
    nc = s.nc
    total = zero(eltype(x))
    @inbounds for j in 1:nc
        sub = zero(eltype(x))
        for i in 1:nc
            sub += A.cc[i,j] * x[i]
        end
        total += x[j] * sub
    end
    @inbounds for k in eachindex(s.ppoffsets)
        blk = s.ncblocks + k
        P = s.blocksizes[blk]
        o = s.ppoffsets[k] - 1
        xo = s.boffsets[blk] - 1
        for j in 1:P
            sub = zero(eltype(x))
            for i in 1:P
                sub += A.pp[o + (j - 1) * P + i] * x[xo + i]
            end
            total += x[xo + j] * sub
        end
        cross = zero(eltype(x))
        for e in s.cpptr[k]:s.cpptr[k+1]-1
            a = s.cpblock[e]
            na = s.blocksizes[a]
            ao = s.boffsets[a] - 1
            co = s.cpoffsets[e] - 1
            for r in 1:na
                sub = zero(eltype(x))
                for j in 1:P
                    sub += A.cp[co + (r - 1) * P + j] * x[xo + j]
                end
                cross += x[ao + r] * sub
            end
        end
        total += 2 * cross
    end
    return total
end

# The full symmetric matrix, in the order of the unknowns (for testing and inspection)
function Base.Matrix(A::SchurBlockMatrix{T}) where T
    s = A.structure
    n = schurdim(A)
    M = zeros(T, n, n)
    M[1:s.nc, 1:s.nc] .= A.cc
    for k in eachindex(s.ppoffsets)
        blk = s.ncblocks + k
        P = s.blocksizes[blk]
        xr = s.boffsets[blk] .+ (0:P-1)
        M[xr, xr] .= reshape(view(A.pp, s.ppoffsets[k] .+ (0:P*P-1)), P, P)
        for e in s.cpptr[k]:s.cpptr[k+1]-1
            a = s.cpblock[e]
            na = s.blocksizes[a]
            ar = s.boffsets[a] .+ (0:na-1)
            B = permutedims(reshape(view(A.cp, s.cpoffsets[e] .+ (0:na*P-1)), P, na))
            M[ar, xr] .= B
            M[xr, ar] .= B'
        end
    end
    return M
end
