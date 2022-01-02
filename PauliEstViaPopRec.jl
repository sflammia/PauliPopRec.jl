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

# Given a list of measurement outcomes R, and a list A of "B adjusted" inputs,
# estimate the probability of all-0 using the estimator with lower variance.
# Here w[k] = (-1/2)^(k-1) / m is the weighing factor for the estimator. 
# This estimator assumes B is not the all-0 vector already. 
function EstProb(B::Vector{Int8},A::Matrix{Int8},R::Matrix{Int8},w::Vector{Float64},t::Integer)::Float64
    AB = star(A, B' .+ zeros(Int8,size(A)) )
    ABR = AB .⊻  R
    w' * (counts(sum(ABR, dims = 2),0:t) - counts(sum(AB, dims = 2),0:t) )
end

# Estimate the whole channel using branch and prune from probe states A
# and measurement record R, with 𝛿 the pruning threshold.
# This implementation has a bad runtime of O(m*n^2/𝛿). We could save a factor of n
# by recycling the marginal estimates from round j and using them in round j+1.
function EstChan(A::Matrix{Int8},R::Matrix{Int8},𝛿::Float64)::Tuple{Vector{Float64}, Matrix{Int8}}
    (m,n) = size(A)
    p = Float64[]
    P = Vector{Int8}[[0],[1],[2],[3]]
    w = (-1/2).^(0:n) / m
    for j = 2:n
        q = Float64[]
        Q = Array{Int8,1}[]
        AA = A[:,1:j]
        RR = R[:,1:j]
        ww = w[1:(j+1)]
        for s = 1:length(P)
            prefx = P[s]
            for c = Int8.(0:3)
                B = vcat(prefx, c)
                est = EstProb(B, AA, RR, ww, j)
                if est > 𝛿
                    q = vcat(q, est)
                    push!(Q,B)
                end
            end
        end
        p = q
        P = Q
    end
    # finally, check the zero string
    est = w' * counts(sum(R, dims = 2),0:n)
    if est > 𝛿
        pushfirst!(p, est)
        pushfirst!(P, zeros(Int8,n))
    end
    (p,vcat(P'...))
end

# represent a sparse Pauli channel as a dictionary
sparseP(p::Vector{Float64},P::Matrix{Int8})::Dict{Vector{Int8},Float64} = 
    Dict{Vector{Int8},Float64}( [P[j,:] for j in 1:size(P)[1] ] .=> p )

# compute the total variation distance between two Pauli dictionaries
tvd(p::Vector{Float64},P::Matrix{Int8},q::Vector{Float64},Q::Matrix{Int8})::Float64 = 
    tvd(sparseP(p,P),sparseP(q,Q))
tvd(pP::Dict{Vector{Int8},Float64},qQ::Dict{Vector{Int8},Float64})::Float64 = 
    sum(abs,values(merge(-,pP,qQ)))/2

# Brute force compute the estimator on *all* the Paulis. Don't use with large n!
EstAll(A::Matrix{Int8},R::Matrix{Int8})::Vector{Float64} = EstAll(A,R,size(A)[1],size(A)[2])
function EstAll(A::Matrix{Int8},R::Matrix{Int8},m::Integer,n::Integer)::Vector{Float64}
    @assert n < 6
    w = (-1/2).^(0:n) / m
    p0 = w' * counts(sum(R, dims = 2),0:n)
    pushfirst!([EstProb(reverse(digits(Int8, j,base=4,pad=n)),A,R,w,n) for j = 1:4^n-1], p0)
end