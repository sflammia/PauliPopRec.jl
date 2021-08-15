# Needed for sampling discrete probability distributions efficiently
using StatsBase

# Uniform sample from the d-simplex; optional 2nd argument makes 1st element 1-eps.
randp(d::Int) = diff([0; sort(rand(d-1)); 1])
randp(d::Int,ε::Float64) = [(1-ε); ε.*randp(d-1)]

# Define a random Pauli channel on n qubits with support size k and
# weight 1-𝛿 on the identity Pauli. Returns a tuple (p,P).
RandomPauliChannel(n::Int,k::Int,𝛿::Float64) = (randp(k,𝛿), vcat(zeros(Int,1,n),rand(0:3,k-1,n)))

# Sample the channel given by (p,P) m times and return the associated Paulis
function SampleChannel(p,P,m::Int)
    (k,n) = size(P)
    w = Weights(p)
    s = sample(1:k, w, m) # sampling
    [ P[a,b] for a in s, b in 1:n ]
end

# This is the star product from the paper, aka the symplectic product.
# If A, B = {0,1,2,3} are single-qubit Paulis, then A(A-B)B == 0 iff A and B commute.
star(A,B) = (B .!== 0) .& (A .!== B) .& (A .!== 0)

# Given the channel (p,P) and a list of states A, record the measurement outcomes R
function Measure(A,p,P)
    m = size(A)[1]
    C = SampleChannel(p,P,m)
    star(A,C)
end

# Given a list of measurement outcomes R, estimate the probability of all-0.
function EstProb0(R)
    hamw = sum(R, dims = 2)
    sum((-1/2).^hamw)/length(hamw)
end

# Same as EstProb0, except we allow a nontrivial pauli B, where B might
# be defined only on the first t qubits.
function EstProb(B,A,R)
    if all(B .== 0)
        return EstProb0(R)
    end
    t = length(B)
    m = size(A)[1]
    AB = star(A[:,1:t], kron(ones(Int,m,1),B'))
    newR = (AB .⊻  R[:,1:t])
    EstProb0(newR) - EstProb0(AB)
end

# Estimate the whole channel using branch and prune from probe states A
# and measurement record R, with 𝛿 the pruning threshold.
# This implementation has a bad runtime of O(m*n^2/𝛿). We could save a factor of n
# by recycling the marginal estimates from round j and using them in round j+1.
function EstChan(A,R,𝛿::Float64)
    n = size(A)[2]
    p = Float64[]
    P = [[0],[1],[2],[3]]
    for j = 1:n-1
        q = Float64[]
        Q = Array{Int,1}[]
        for s = 1:length(P)
            prefx = P[s]
            for c = 0:3
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
function sparseP(p,P)
    n = size(P)[2]
    num = (4 .^ (n-1:-1:0))
    sparsevec((P * num) .+ 1, p, 4^n)
end

# compute the total variation distance between two sparse Pauli channels
tvd(p,P,q,Q) = tvd(sparseP(p,P),sparseP(q,Q))
tvd(pP,qQ) = sum(abs.(qQ-pP))/2

# Brute force compute the estimator on *all* the Paulis. Don't use with large n!
EstAll(A,R) = EstAll(A,R,size(A)[2])
EstAll(A,R,n::Int) = [EstProb(reverse(digits(j,base=4,pad=n)),A,R) for j = 0:4^n-1]
