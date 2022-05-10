module PauliPopRec

# Needed for sampling discrete probability distributions efficiently
using StatsBase: counts, std, weights, sample 

export randpaulichannel, probepaulichannel, pauli_indrec, pauli_poprec, pauli_stderr, pauli_tvd, pauli_totalrecall

# Define an alias type for Pauli channels
const PauliChannel = Dict{Vector{Int8},Float64}

# load utility functions
include("utils.jl")

"""
    randpaulichannel(n::Integer, k::Integer[,𝛿::Float64])

Output a dictionary P => p where the values p are the Pauli error rates and the keys P are Int8 vectors with n elements in {0,1,2,3}. If the optional 
third argument is set, then the all-0 vector will have probaility (1-𝛿).

# Examples
Here is a two-qubit channel with probability .9 of the identity, and with YI and ZZ errors. 
```
julia> randpaulichannel(2, 3, .1)
Dict{Vector{Int8}, Float64} with 3 entries:
  [0, 0] => 0.9
  [2, 0] => 0.0832336
  [3, 3] => 0.0167664
```
"""
function randpaulichannel(n::Integer, k::Integer)::PauliChannel
    @assert 1 ≤ k ≤ 4^n "Pauli channels have at most 4^n error rates."
    P = unique(rand(Int8.(0:3),k,n), dims = 1)
    # keep adding elements until the number of unique rows in P is k.
    # this is slow if k is very large.
    while size(P)[1] < k
        P = unique(vcat(P,rand(Int8.(0:3),1,n)), dims = 1)
    end
    Dict( [P[i,:] for i=1:k] .=> randp(k) )
end

function randpaulichannel(n::Integer, k::Integer, 𝛿::Float64)::PauliChannel
    @assert 0 ≤ 𝛿 < 1 "Identity probability 𝛿 should be in [0,1)."
    @assert 2 ≤ k ≤ 4^n "Pauli channels have at most 4^n error rates."
    P = unique(vcat(zeros(Int8,1,n),rand(Int8.(0:3),k-1,n)),dims=1)
    # keep adding elements until the number of unique rows in P is k.
    # this is slow if k is very large.
    while size(P)[1] < k
        P = unique(vcat(P,rand(Int8.(0:3),1,n)), dims = 1)
    end    
    Dict( [P[i,:] for i=1:k] .=> randp(k,𝛿) )
end


"""
    probepaulichannel(P::PauliChannel, A::Matrix{<:Integer})

Given the Pauli channel P and an (m,n) matrix of n-qubit Paulis A, sample the measurement outcomes R that result from sending the +1  eigenstates of A into the channel. 
The output is a 0/1 matrix of type Int8 with the same dimensions of A recording where the output was flipped (0) or not (1) by the sampled channel action. 

# Examples
Prepare the state |00⟩ three times and the state |++⟩ three times (this is A). 
Then send these through a Pauli channel P that either applies XZ or XX. 
The first bit flips always with the input |00⟩, and never flips with the |++⟩ state since it is stabilized by XX. 
The second bit is non-deterministic since it could be affected by an X or a Z on each draw.
```
julia> A = Int8[3 3; 3 3; 3 3; 1 1; 1 1; 1 1]; 

julia> P = PauliChannel( Int8[1; 3] => 0.5, Int8[1; 1] => 0.5)
Dict{Vector{Int8}, Float64} with 2 entries:
  [1, 3] => 0.5
  [1, 1] => 0.5

julia> probepaulichannel(P, A)
6×2 Matrix{Int8}:
 1  1
 1  1
 1  0
 0  1
 0  0
 0  1
```
"""
function probepaulichannel(P::PauliChannel, A::Matrix{<:Integer})::Matrix{Int8}
    m = size(A)[1]
    C = samplepaulichannel(P,m)
    star(A,C)
end


"""
    pauli_indrec(B::Vector{<:Integer}, A::Matrix{<:Integer}, R::Matrix{<:Integer}[,v::Float64 = 0.0])

Use "Individual Recovery" to estimate the probability of the Pauli B from a list of input states A, measurement results R. 

The optional argument v defaults to 0 and quantifies the heralded error rate. 

This function estimates the probability of B using the low-variance estimator
described in Prop. 27 of arXiv:2105.02885.

# Examples
Initialize a random Pauli channel on 3 qubits with support on 3 outcomes and p[1] = 0.8. 
The nontrivial outcomes are YXX and ZXX. 
If we prepare 10^4 random probe states (A) and measure the output (R), 
then we estimate  for the probability of the YXX error.
This is correct to about 1%, as we might hope from this many samples. 
```
julia> n = 3; P = randpaulichannel(n, 3, 0.2)
Dict{Vector{Int8}, Float64} with 3 entries:
  [2, 1, 1] => 0.14533
  [3, 1, 1] => 0.0546704
  [0, 0, 0] => 0.8

julia> m = 10^4; A = rand(Int8.(1:3),m,n); # sample 10^4 random inputs

julia> R = probepaulichannel(P,A); # Send the probe states A through the channel and measure.

julia> B = collect(keys(P))[1] # get the YXX error
3-element Vector{Int8}:
 2
 1
 1

julia> pauli_indrec(B,A,R) # estimate p(YXX) for the channel P.
0.1473375
```
"""
function pauli_indrec(B::Vector{<:Integer}, A::Matrix{<:Integer}, R::Matrix{<:Integer}, v::Float64=0.0)::Float64
    (m,t) = size(A)
    @assert size(R) == (m,t) "Size of R and A should match."
    @assert length(B) == t "Length of B should match size(A)[2]."
    @assert 0 ≤ v ≤ 1/4 "Erasure rate v should be in [0,1/4]."
    h = (2*v+1)/(2*v-2); # h = -1/2 when v = 0. Larger v for hearlded failures.
    w = h.^(0:t) / m
    Ax, Az = bits(A)
    individual_recovery(B,Ax,Az,R,w,t)
end


"""
    pauli_poprec(A::Matrix{<:Integer}, R::Matrix{<:Integer}, 𝛿::Float64[, v::Float64 = 0.0])

Estimate the whole channel using the branch and prune from probe states A
and measurement record R, with 𝛿 the pruning threshold. 
The output is a Pauli channel (a dictionary from Vector{Int8} to Float64.) 
The optional final argument adds a heralded failure rate v. 

*NOTE:* This implementation has a bad runtime of O(m*n^2/𝛿). We could save a factor of n
by recycling the marginal estimates from round j and using them in round j+1.

# Examples
Here is an example with n = 4 qubits and a Pauli channel with only YXZI and ZZYY nontrivial errors. 
We call pauli_poprec with a pruning threshold of 1/sqrt(m) where m is the number of measurements. 
The algorithm recovers an estimate of the channel with total variation distance about 0.015%.
```
julia> n = 4;  P = randpaulichannel(n, 3, 0.1) # create a random Pauli channel.
Dict{Vector{Int8}, Float64} with 3 entries:
  [2, 1, 3, 0] => 0.0390184
  [3, 3, 2, 2] => 0.0609816
  [0, 0, 0, 0] => 0.9

julia> m = 10^6; A = rand(Int8.(1:3),m,n); # sample 10^6 random inputs

julia> R = probepaulichannel(P, A); # Send the probe states A through the channel and measure.

julia> Q = pauli_poprec(A, R, 1/sqrt(m)) # estimate the channel using Population Recovery
Dict{Vector{Int8}, Float64} with 3 entries:
  [2, 1, 3, 0] => 0.0389027
  [3, 3, 2, 2] => 0.0611194
  [0, 0, 0, 0] => 0.899938

julia> pauli_tvd(P,Q)
0.00015791612989364545
```
"""
function pauli_poprec(A::Matrix{<:Integer}, R::Matrix{<:Integer}, 𝛿::Float64,v::Float64=0.0)::PauliChannel
    @assert 0 ≤ v ≤ 1/4 "Erasure rate v should be in [0,1/4]."
    @assert size(A) == size(R) "A and R should have the same size."
    P = PauliChannel(Vector{Int8}[] => 0.0)
    (m,n) = size(A)
    h = (2*v+1)/(2*v-2); # h = -1/2 when v = 0. Larger v for hearlded failures.
    w = h.^(0:n) / m
    Ax, Az = bits(A)
    for j = 1:n
        Q = PauliChannel()
        AAx = Ax[:,1:j]
        AAz = Az[:,1:j]
        RR = R[:,1:j]
        ww = w[1:(j+1)]
        for prefx in keys(P)
            for c = Int8.(0:3)
                B = vcat(prefx, c)
                est = individual_recovery(B, AAx, AAz, RR, ww, j)
                if est > 𝛿
                    Q[B] = est
                end
            end
        end
        P = Q
    end
    P
end


"""
    pauli_stderr(P::PauliChannel, A::Matrix{<:Integer}, R::Matrix{<:Integer}[,v::Float64 = 0.0])

Estimate 1 std error bars using the jackknife estimator for variance. 
The optional final argument adds a heralded error rate v.

# Examples
Here is an example with n = 4 qubits and a Pauli channel with only YXZI and ZZYY nontrivial errors. 
We call pauli_poprec with a pruning threshold of 1/sqrt(m) where m is the number of measurements. 
The error bars on each individual error are computed from the estimated support Q.
```
julia> n = 4;  P = randpaulichannel(n, 3, 0.1) # create a random Pauli channel.
Dict{Vector{Int8}, Float64} with 3 entries:
  [2, 1, 3, 0] => 0.0390184
  [3, 3, 2, 2] => 0.0609816
  [0, 0, 0, 0] => 0.9

julia> m = 10^6; A = rand(Int8.(1:3),m,n); # sample 10^6 random inputs

julia> R = probepaulichannel(P,A); # Send the probe states A through the channel and measure.

julia> Q = pauli_poprec(A,R,1/sqrt(m)) # estimate the channel using Population Recovery
Dict{Vector{Int8}, Float64} with 3 entries:
  [2, 1, 3, 0] => 0.0389027
  [3, 3, 2, 2] => 0.0611194
  [0, 0, 0, 0] => 0.899938

julia> pauli_stderr(Q, A, R)
3-element Vector{Float64}:
 0.00023138914241481277
 0.0002582473008160544
 0.00031401841511195555
```
"""
function pauli_stderr(P::PauliChannel, A::Matrix{<:Integer}, R::Matrix{<:Integer}, v::Float64=0.0)::Vector{Float64}
    @assert 0 ≤ v ≤ 1/4 "Erasure rate v should be in [0,1/4]."
    Q = vcat(collect(keys(P))'...)
    h = (2*v+1)/(2*v-2); # h = -1/2 when v = 0. Larger v for hearlded failures.
    k = size(Q)[1]
    (m,n) = size(A)
    s = zeros(k)
    Ax, Az = bits(A)
    for j = 1:k
        AQ = star(Q[j,:],Ax,Az) 
        s[j] = std(dropdims(h.^sum(AQ .⊻  R,dims=2) - h.^sum(AQ,dims=2),dims=2))
    end
    s/sqrt(m)
end


"""
    pauli_tvd(P::PauliChannel, Q::PauliChannel)

Compute the total variation distance between two Pauli channels input as dictionaries.

# Examples
Two Pauli channels each with 0.7 chance of no error, but otherwise disjoint error supports. 
Here the nontrivial errors are YIX for P, and IXX for Q. 
The total variation distance should be 0.3, and it is.
```
julia> P = randpaulichannel(3, 2, 0.3);  Q = randpaulichannel(3, 2, 0.3);

julia> keys(P), keys(Q)
(Vector{Int8}[[0, 0, 0], [2, 0, 1]], Vector{Int8}[[0, 0, 0], [0, 1, 1]])

julia> pauli_tvd(P,Q)
0.3
```
"""
pauli_tvd(P::PauliChannel,Q::PauliChannel)::Float64 =
    sum(abs,values(merge(-,P,Q)))/2


"""
    pauli_totalrecall(A::Matrix{Int8},R::Matrix{Int8}[,v::Float64 = 0.0])

Total recovery of all Pauli error rates whose individual recovery estimator is positive. 
(I.e., "Total Rec. All")
This is not the same as calling pauli_poprec(A,R,0.0) since it never does pruning, it simply applies individual recovery to all 4^n possible Pauli error rates and returns the positive esimates. 

This is a very expensive function: *Don't use with large n!*

The optional last argument adds a heralded failure rate v.

# Examples
Here is an example with n = 2 qubits. 
The pauli_totalrecall function identifies the two correctly supported IZ and ZX errors, 
but also returns positive estimates for several other spurious errors with 
suspiciously low estimated probabilities. 
```
julia> n = 2;  P = randpaulichannel(n, 3, 0.1) # create a random Pauli channel.
Dict{Vector{Int8}, Float64} with 3 entries:
  [0, 0] => 0.9
  [2, 0] => 0.0118879
  [0, 2] => 0.0881121

julia> m = 10^6; A = rand(Int8.(1:3), m, n); # sample 10^6 random inputs

julia> R = probepaulichannel(P, A); # Probe the channel P with states A.

julia> Q = pauli_totalrecall(A, R) # find all positive estimates
Dict{Vector{Int8}, Float64} with 9 entries:
  [0, 0] => 0.900109
  [2, 1] => 0.0001725
  [1, 3] => 5.25e-6
  [0, 2] => 0.0881115
  [2, 0] => 0.012216
  [1, 0] => 0.00025875
  [3, 3] => 8.925e-5
  [3, 2] => 0.00051975
  [0, 1] => 8.4e-5
```
"""
function pauli_totalrecall(A::Matrix{<:Integer}, R::Matrix{<:Integer}, v::Float64=0.0)::PauliChannel
    (m,n) = size(A)
    @assert 0 ≤ v ≤ 1/4 "Erasure rate v should be in [0,1/4]."
    h = (2*v+1)/(2*v-2); # h = -1/2 when v = 0. Larger v for hearlded failures.
    w = h.^(0:n) / m
    p0 = w' * counts(sum(R, dims = 2),0:n)
    Ax, Az = bits(A)
    s = pushfirst!([individual_recovery(reverse(digits(Int8, j,base=4,pad=n)),Ax,Az,R,w,n) for j = 1:4^n-1], p0)
    S = vcat(reverse.(digits.(Int8, findall(x -> x > 0, s) .- 1, base=4,pad=n))'...)
    s = s[s .> 0]
    Dict( [ S[i,:] for i=1:length(s)] .=> s )
end

end # module