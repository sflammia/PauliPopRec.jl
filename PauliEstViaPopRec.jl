# Needed for sampling discrete probability distributions efficiently
using StatsBase

# Uniform sample from the d-simplex; optional 2nd argument makes 1st element 1-eps.
randp(d::Integer)::Vector{Float64} = diff([0; sort(rand(d-1)); 1])
randp(d::Integer,ε::Float64)::Vector{Float64} = [(1-ε); ε.*randp(d-1)]

# Define a random Pauli channel on n qubits with support size at most k and
# weight 1-𝛿 on the identity Pauli. Returns a tuple (p,P).
function RandomPauliChannel(n::Integer,k::Integer,𝛿::Float64)::Tuple{Vector{Float64}, Matrix{Int8}}
    P = unique(vcat(zeros(Int8,1,n),rand(Int8.(0:3),k-1,n)),dims=1)
    (randp(size(P)[1],𝛿), P)
end

# Sample the channel given by (p,P) m times and return the associated Paulis
function SampleChannel(p::Vector{Float64},P::Matrix{Int8},m::Integer)::Matrix{Int8}
    (k,n) = size(P)
    w = Weights(p)
    s = sample(1:k, w, m) # sampling
    [ P[a,b] for a in s, b in 1:n ]
end

# This is the star product from the paper, aka the symplectic product.
# If A, B = {0,1,2,3} are single-qubit Paulis, then star(A,B) == 0 iff A and B commute.
star(A::Matrix{Int8},B::Matrix{Int8})::Matrix{Int8} = ((A .>> 1) .& B) .⊻ ((B .>> 1) .& A)

# Given the channel (p,P) and a list of states A, record the measurement outcomes R
function Measure(A::Matrix{Int8},p::Vector{Float64},P::Matrix{Int8})::Matrix{Int8}
    m = size(A)[1]
    C = SampleChannel(p,P,m)
    star(A,C)
end

# Given a list of measurement outcomes R, estimate the probability of all-0.
EstProb0(R::Matrix{Int8})::Float64 = (counts(sum(R, dims = 2),0:size(R)[2])' * (-1/2).^(0:size(R)[2]) )/size(R)[1]

# Same as EstProb0, except we allow a nontrivial pauli B, where B might
# be defined only on the first t qubits.
function EstProb(B::Vector{Int8},A::Matrix{Int8},R::Matrix{Int8})::Float64
    if all(B .== 0)
        return EstProb0(R)
    end
    t = length(B)
    m = size(A)[1]
    AB = star(A[:,1:t], B' .+ zeros(Int8,(m,t)) )
    newR = AB .⊻  R[:,1:t]
    EstProb0(newR) - EstProb0(AB)
end

# Estimate the whole channel using branch and prune from probe states A
# and measurement record R, with 𝛿 the pruning threshold.
# This implementation has a bad runtime of O(m*n^2/𝛿). We could save a factor of n
# by recycling the marginal estimates from round j and using them in round j+1.
function EstChan(A::Matrix{Int8},R::Matrix{Int8},𝛿::Float64)::Tuple{Vector{Float64}, Matrix{Int8}}
    n = size(A)[2]
    p = Float64[]
    P = Vector{Int8}[[0],[1],[2],[3]]
    for j = 1:n-1
        q = Float64[]
        Q = Array{Int8,1}[]
        for s = 1:length(P)
            prefx = P[s]
            for c = Int8.(0:3)
                B = vcat(prefx, c)
                est = EstProb(B, A, R)
                if est > 𝛿
                    q = vcat(q, est)
                    push!(Q,B)
                end
            end
        end
        p = q
        P = Q
    end
    (p,vcat(P'...))
end

# needed for representing Pauli channels as sparse vectors
using SparseArrays

# represent the channel as a sparse array by converting each Pauli to an Int
function sparseP(p::Vector{Float64},P::Matrix{<:Integer})::SparseVector{Float64, Int64}
    n = size(P)[2]
    num = (4 .^ (n-1:-1:0))
    sparsevec((P * num) .+ 1, p, 4^n)
end

# compute the total variation distance between two sparse Pauli channels
tvd(p::Vector{Float64},P::Matrix{Int8},q::Vector{Float64},Q::Matrix{Int8})::Float64 = tvd(sparseP(p,P),sparseP(q,Q))
tvd(pP,qQ)::Float64 = sum(abs,qQ-pP)/2

# Brute force compute the estimator on *all* the Paulis. Don't use with large n!
EstAll(A,R) = EstAll(A,R,size(A)[2])
EstAll(A,R,n::Integer) = [EstProb(reverse(digits(j,base=4,pad=n)),A,R) for j = 0:4^n-1]
