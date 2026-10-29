using StaticArrays, Static, SparseArrays, LinearAlgebra

# The Schur-complement linear system. The unfixed variables are split by the iseliminable trait into
# retained (c) and eliminable (p) variables, and the normal equations
#     [H_cc  H_cp] [x_c]   [g_c]
#     [H_pc  H_pp] [x_p] = [g_p]
# are solved as
#     S = H_cc - H_cp H_pp⁻¹ H_pc,   S x_c = g_c - H_cp H_pp⁻¹ g_p,   x_p = H_pp⁻¹ (g_p - H_pc x_c),
# with H_pp block diagonal (one small static block per eliminable variable), so the only dense
# factorization is of S, whose size is the total dimension of the retained variables.

# Whether a problem's unfixed variables call for the Schur-complement linear system
function useschur(variables, unfixed)
    anyeliminable = false
    anyretained = false
    for (i, unfixed_) in enumerate(unfixed)
        if unfixed_
            if dynamic(iseliminable(@inbounds variables[i]))
                anyeliminable = true
            else
                anyretained = true
            end
        end
    end
    return anyeliminable && anyretained
end

"""
    MultiVariateLSschur(problem::NLLSProblem, unfixed)

Construct the Schur-complement linear system of `problem` for the unfixed variables `unfixed`
(a boolean vector), eliminating the variables whose type is [`iseliminable`](@ref). Throws an
`ArgumentError` if a cost depends on two unfixed eliminable variables, or if an eliminable variable
is not of static size.
"""
function MultiVariateLSschur(problem::NLLSProblem, unfixed::AbstractVector{Bool})
    variables = problem.variables
    nv = length(variables)
    @assert length(unfixed) == nv
    eliminable = [unfixed[i] && dynamic(iseliminable(variables[i])) for i in 1:nv]
    retainedind = findall(i -> unfixed[i] && !eliminable[i], 1:nv)
    eliminableind = findall(eliminable)
    isempty(retainedind) && throw(ArgumentError("The Schur-complement linear system needs at least one retained (non-eliminable) unfixed variable"))
    isempty(eliminableind) && throw(ArgumentError("The Schur-complement linear system needs at least one eliminable unfixed variable"))
    for i in eliminableind
        is_static(nvars(variables[i])) isa True || throw(ArgumentError("Eliminable variable $i ($(typeof(variables[i]))) must have a static size (nvars returning a StaticInt)"))
    end

    # Order the blocks: retained ones in variable order, then eliminable ones grouped by size
    elimsizes = [Int(dynamic(nvars(variables[i]))) for i in eliminableind]
    order = sortperm(elimsizes; alg=Base.Sort.DEFAULT_STABLE)
    eliminableind = eliminableind[order]
    elimsizes = elimsizes[order]
    Ps = Tuple(unique(elimsizes))
    groups = Tuple(findfirst(==(P), elimsizes):findlast(==(P), elimsizes) for P in Ps)
    ncblocks = length(retainedind)
    neblocks = length(eliminableind)
    blockindices = zeros(UInt, nv)
    blocksizes = Vector{Int}(undef, ncblocks + neblocks)
    for (b, i) in enumerate(retainedind)
        blockindices[i] = b
        blocksizes[b] = Int(dynamic(nvars(variables[i])))
    end
    for (k, i) in enumerate(eliminableind)
        blockindices[i] = ncblocks + k
        blocksizes[ncblocks + k] = elimsizes[k]
    end
    boffsets = computestartindices(blocksizes)
    nc = boffsets[ncblocks + 1] - 1
    n = boffsets[end] + blocksizes[end] - 1
    ppoffsets = Vector{Int}(undef, neblocks)
    off = 1
    for k in 1:neblocks
        ppoffsets[k] = off
        off += elimsizes[k] ^ 2
    end
    pplen = off - 1

    # Find the retained blocks coupled to each eliminable block: collect the (eliminable, retained)
    # pairs of every cost, then sort them into a CSC pattern over the eliminable blocks
    varcostmap = getvarcostmap(problem)
    pairs = Tuple{Int, Int}[]
    for c in 1:size(varcostmap, 2)
        rows = view(rowvals(varcostmap), nzrange(varcostmap, c))
        e = 0
        evar = 0
        for v in rows
            b = Int(blockindices[v])
            if b > ncblocks
                if e != 0 && e != b
                    throw(ArgumentError("Cost $c depends on the two eliminable variables $evar and $v; the Schur-complement linear system requires each cost to depend on at most one unfixed eliminable variable"))
                end
                e = b
                evar = v
            end
        end
        if e != 0
            for v in rows
                b = Int(blockindices[v])
                if 0 < b <= ncblocks
                    push!(pairs, (e - ncblocks, b))
                end
            end
        end
    end
    unique!(sort!(pairs))
    cpptr = zeros(Int, neblocks + 1)
    cpblock = Vector{Int}(undef, length(pairs))
    cpoffsets = Vector{Int}(undef, length(pairs))
    off = 1
    for (i, (k, a)) in enumerate(pairs)
        cpptr[k + 1] += 1
        cpblock[i] = a
        cpoffsets[i] = off
        off += blocksizes[a] * elimsizes[k]
    end
    cpptr[1] = 1
    cumsum!(cpptr, cpptr)
    cplen = off - 1

    structure = SchurStructure{length(Ps)}(nc, ncblocks, blocksizes, boffsets, ppoffsets, cpptr, cpblock, cpoffsets, groups)
    A = SchurBlockMatrix{Float64, length(Ps)}(zeros(nc, nc), zeros(pplen), zeros(cplen), structure)
    x = MultiVariateLSschur_x{Ps, length(Ps)}(zeros(n), blockindices, convert(Vector{UInt}, boffsets), structure, zeros(nc, nc), zeros(nc, nc), zeros(nc), zeros(pplen), zeros(cplen))
    return LinearSystem(x, MultiVariateLSschur_ls{length(Ps)}(A, zeros(n)))
end

# Add the block a[ri, rj] of a cost's Hessian at block (bi, bj) of A, and its transpose at (bj, bi)
@inline function addschurblock!(A::SchurBlockMatrix, bi::Int, bj::Int, a, ri, rj)
    s = A.structure
    ncb = s.ncblocks
    if bi <= ncb && bj <= ncb
        # Both retained
        oi = @inbounds s.boffsets[bi] - 1
        oj = @inbounds s.boffsets[bj] - 1
        cc = A.cc
        @inbounds for q in eachindex(rj)
            for r in eachindex(ri)
                v = a[ri[r], rj[q]]
                cc[oi + r, oj + q] += v
                if bi != bj
                    cc[oj + q, oi + r] += v
                end
            end
        end
    elseif bi > ncb && bj > ncb
        bi == bj || throw(ArgumentError("A cost couples two eliminable variables"))
        k = bi - ncb
        P = length(ri)
        o = @inbounds s.ppoffsets[k] - 1
        pp = A.pp
        @inbounds for q in 1:P
            for r in 1:P
                pp[o + (q - 1) * P + r] += a[ri[r], ri[q]]
            end
        end
    else
        # One retained (c) and one eliminable (k)
        if bi <= ncb
            c, k, rc, rp, transposed = bi, bj - ncb, ri, rj, false
        else
            c, k, rc, rp, transposed = bj, bi - ncb, rj, ri, true
        end
        e = @inbounds s.cpptr[k]
        @inbounds while s.cpblock[e] != c
            e += 1
        end
        o = @inbounds s.cpoffsets[e] - 1
        P = length(rp)
        cp = A.cp
        @inbounds for r in eachindex(rc)
            for q in 1:P
                cp[o + (r - 1) * P + q] += transposed ? a[rp[q], rc[r]] : a[rc[r], rp[q]]
            end
        end
    end
    return nothing
end

function updatesymschur!(A::SchurBlockMatrix, a, vars, varflags, blockindices)
    loffseti = static(0)
    @unroll for i in 1:MAX_ARGS
        if i <= length(vars) && bitiset(varflags, i)
            nvi = nvars(vars[i])
            rangei = SR(1, nvi) .+ loffseti
            loffseti += nvi
            bi = Int(@inbounds blockindices[i])
            addschurblock!(A, bi, bi, a, rangei, rangei)

            loffsetj = static(0)
            @unroll for j in 1:i-1
                if bitiset(varflags, j)
                    @inbounds nvj = nvars(vars[j])
                    rangej = SR(1, nvj) .+ loffsetj
                    loffsetj += nvj
                    addschurblock!(A, bi, Int(@inbounds blockindices[j]), a, rangei, rangej)
                end
            end
        end
    end
end

function updatesymlinearsystem!(linsystem::MultiVariateLSschur, g::AbstractVector, H::AbstractArray, vars, varflags, blockindices)
    updateb!(getb(linsystem), g, vars, varflags, linsystem.x.boffsets, blockindices)
    updatesymschur!(getA(linsystem), H, vars, varflags, blockindices)
end

# Damping: both partitions are damped, in the order of the unknowns
function maxabsdiag(A::SchurBlockMatrix)
    m = 0.0
    s = A.structure
    @inbounds for i in 1:s.nc
        m = max(m, abs(A.cc[i,i]))
    end
    @inbounds for k in eachindex(s.ppoffsets)
        P = s.blocksizes[s.ncblocks + k]
        o = s.ppoffsets[k]
        for d in 1:P
            m = max(m, abs(A.pp[o + (d - 1) * (P + 1)]))
        end
    end
    return m
end

function updatescaling!(lmd::AnyLevenbergData, A::SchurBlockMatrix)
    if lmd.lambda == 0
        lmd.lambda = maxabsdiag(A) * 1e-6
    end
end

function updatescaling!(lmd::AnyMarquardtData, A::SchurBlockMatrix)
    if lmd.lambda == 0
        lmd.lambda = 1e-6
    end
    scaling = lmd.scaling
    mapdiag!((i, v) -> (@inbounds scaling[i] = max(scaling[i], abs(v)); v), A)
    m = maximum(scaling)
    scaling .= max.(scaling, m * 1e-8)
end

dampenhessian!(A::SchurBlockMatrix, ::AnyLevenbergData, dlambda) = uniformscaling!(A, dlambda)
function dampenhessian!(A::SchurBlockMatrix, lmd::AnyMarquardtData, dlambda)
    scaling = lmd.scaling
    mapdiag!((i, v) -> v + dlambda * @inbounds(scaling[i]), A)
end

# The inverse of a small symmetric positive (semi-)definite block
@inline function invsymblock(H::SMatrix{P, P, T}) where {P, T}
    cholfac = StaticArrays._cholesky(Size(H), H, false)
    if issuccess(cholfac)
        return cholfac \ SMatrix{P, P, T}(I)
    end
    Hinv = inv(H)
    return all(isfinite, Hinv) ? Hinv : pinv(H)
end

@inline loadsvector(v, o, ::Val{P}) where P = @inbounds SVector{P, Float64}(ntuple(j -> v[o + j], Val(P)))
@inline function storesvector!(v, o, x::SVector{P}) where P
    @inbounds for j in 1:P
        v[o + j] = x[j]
    end
end

# Eliminate one group of eliminable blocks of size P: accumulate S and its right-hand side, store
# inv(H_pp) and H_cp inv(H_pp), and put inv(H_pp) g_p in x_p
function reducegroup!(::Val{P}, xs, A, b, x, blocks) where P
    s = xs.structure
    S = xs.S
    rc = xs.rc
    w = xs.w
    cp = A.cp
    @inbounds for k in blocks
        blk = s.ncblocks + k
        o = s.ppoffsets[k] - 1
        H = SMatrix{P, P, Float64}(ntuple(i -> A.pp[o + i], Val(P * P)))
        Hinv = invsymblock(H)
        for i in 1:P*P
            xs.ppinv[o + i] = Hinv[i]
        end
        xo = s.boffsets[blk] - 1
        z = Hinv * loadsvector(b, xo, Val(P))
        storesvector!(x, xo, z)
        erange = s.cpptr[k]:s.cpptr[k+1]-1
        for e in erange
            a = s.cpblock[e]
            ao = s.boffsets[a] - 1
            co = s.cpoffsets[e] - 1
            for r in 1:s.blocksizes[a]
                row = loadsvector(cp, co + (r - 1) * P, Val(P))
                storesvector!(w, co + (r - 1) * P, Hinv * row)
                rc[ao + r] -= dot(row, z)
            end
        end
        for e in erange
            a = s.cpblock[e]
            ao = s.boffsets[a] - 1
            wo = s.cpoffsets[e] - 1
            for e2 in erange
                a2 = s.cpblock[e2]
                ao2 = s.boffsets[a2] - 1
                co2 = s.cpoffsets[e2] - 1
                for r2 in 1:s.blocksizes[a2]
                    row2 = loadsvector(cp, co2 + (r2 - 1) * P, Val(P))
                    for r in 1:s.blocksizes[a]
                        S[ao + r, ao2 + r2] -= dot(loadsvector(w, wo + (r - 1) * P, Val(P)), row2)
                    end
                end
            end
        end
    end
    return nothing
end

# Back-substitute one group: x_p = inv(H_pp) g_p - (H_cp inv(H_pp))' x_c
function backsubstitutegroup!(::Val{P}, xs, x, blocks) where P
    s = xs.structure
    w = xs.w
    @inbounds for k in blocks
        xo = s.boffsets[s.ncblocks + k] - 1
        acc = loadsvector(x, xo, Val(P))
        for e in s.cpptr[k]:s.cpptr[k+1]-1
            a = s.cpblock[e]
            ao = s.boffsets[a] - 1
            wo = s.cpoffsets[e] - 1
            for r in 1:s.blocksizes[a]
                acc -= x[ao + r] * loadsvector(w, wo + (r - 1) * P, Val(P))
            end
        end
        storesvector!(x, xo, acc)
    end
    return nothing
end

@generated function reducegroups!(xs::MultiVariateLSschur_x{Ps, G}, A, b, x) where {Ps, G}
    return Expr(:block, [:(reducegroup!(Val($(Ps[g])), xs, A, b, x, xs.structure.groups[$g])) for g in 1:G]..., :(nothing))
end

@generated function backsubstitutegroups!(xs::MultiVariateLSschur_x{Ps, G}, x) where {Ps, G}
    return Expr(:block, [:(backsubstitutegroup!(Val($(Ps[g])), xs, x, xs.structure.groups[$g])) for g in 1:G]..., :(nothing))
end

"""
    NLLSsolver.schurcomplement!(linsystem::MultiVariateLSschur) -> S

Form the reduced system `S = H_cc - H_cp H_pp⁻¹ H_pc` of the linear system's current (possibly
damped) Hessian, over the retained variables in the order of the unknowns, and return it. For the
undamped Gauss-Newton Hessian at an optimum, `inv(S)` is the covariance of the retained variables
with the eliminated ones marginalized out.
"""
function schurcomplement!(ls::MultiVariateLSschur)
    xs = ls.x
    A = getA(ls)
    b = getb(ls)
    nc = xs.structure.nc
    copyto!(xs.S, A.cc)
    copyto!(xs.rc, 1, b, 1, nc)
    reducegroups!(xs, A, b, getx(ls))
    return xs.S
end

function solve!(ls::MultiVariateLSschur)
    xs = ls.x
    x = getx(ls)
    nc = xs.structure.nc
    S = schurcomplement!(ls)
    # Solve the reduced system
    copyto!(xs.Sfac, S)
    copyto!(x, 1, xs.rc, 1, nc)
    xc = view(x, 1:nc)
    _, info = LAPACK.potrf!('U', xs.Sfac)
    if info == 0
        LAPACK.potrs!('U', xs.Sfac, xc)
    else
        # Not positive definite (e.g. an undamped, rank-deficient system): fall back as try_cholesky! does
        xc .= qr(S) \ view(xs.rc, 1:nc)
    end
    # Back-substitute the eliminated variables
    backsubstitutegroups!(xs, x)
    return x
end
