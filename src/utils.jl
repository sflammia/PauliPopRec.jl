"""
    randp(d::Integer[, ϵ::Float64])

Uniform sample from the d-dimentional probability simplex.
The optional second argument sets the first element to 1-ϵ and 
samples the rest from the (d-1)-dimensional simplex, rescaling down by ϵ.

# Examples
```
julia> randp(3)
3-element Vector{Float64}:
 0.06311863312174104
 0.02572774432010183
 0.9111536225581571

julia> randp(4,.1)
4-element Vector{Float64}:
 0.9
 0.03843804932363781
 0.024716554190428897
 0.03684539648593329
```
"""
randp(d::Integer) = diff([0; sort(rand(d-1)); 1])
randp(d::Integer,ϵ::Float64) = [(1-ϵ); ϵ.*randp(d-1)]


"""
    samplepaulichannel(P::PauliChannel,m::Integer)

Sample m times from the Pauli channel P and return the associated Pauli strings as an Int8 matrix.

# Examples
Here we sample five times from a two-qubit channel that has YZ, XX, and ZY errors. We get the samples YZ, ZY, XX, ZY, ZY. 
```
julia> P = randpaulichannel(2, 3)
Dict{Vector{Int8}, Float64} with 3 entries:
  [2, 3] => 0.272415
  [1, 1] => 0.294461
  [3, 2] => 0.433124

julia> samplepaulichannel(P,5)
5×2 Matrix{Int8}:
 2  3
 3  2
 1  1
 3  2
 3  2
```
"""
function samplepaulichannel(P::PauliChannel,m::Integer)::Matrix{Int8}
    w = weights(collect(values(P)))
    s = sample(1:length(w), w, m) # sampling
    vcat(collect(keys(P))[s]'...)
end


"""
    star(A::Array{<:Integer},B::Array{<:Integer})

This is the star product from the paper, aka the symplectic product. 
If A, B = {0,1,2,3} are single-qubit Paulis, then star(A,B) == 0 iff A and B commute.

The function takes matrices A and B and takes the point-wise symplectic product of the elements. 
*No test is done* to check if A and B have the same size, or if the elements are in {0,1,2,3}. 

# Examples
The input Paulis A = IXYZ and B = XYZI are tested to see if they commute point-wise. 
Only the first and last entries commute; the middle two anticommute.
```
julia> A = Int8[0 1 2 3]; B = Int8[1 2 3 0];

julia> star(A,B)
1×4 Matrix{Int8}:
 0  1  1  0
```
"""
star(A::Array{<:Integer},B::Array{<:Integer})::Array{Int8} = ((A .>> 1) .& B) .⊻ ((B .>> 1) .& A)


"""
    function individual_recovery(B::Vector{<:Integer}, A::Matrix{<:Integer}, R::Matrix{<:Integer}, w::Vector{Float64}, n::Integer)

Computes the individual recovery estimate for the Pauli B from the probe states A, the results R, the weight vector w, and the number of qubits n.

This individual recovery function does not compute w or n for speed reasons.  
This gets called in the inner loop of pauli_poprec, so speed matters.
"""
function individual_recovery(B::Vector{<:Integer}, A::Matrix{<:Integer}, R::Matrix{<:Integer}, w::Vector{Float64}, n::Integer)::Float64
    if all(B .== 0)
        return w' * (counts(sum(R, dims = 2),0:n))
    end
    AB = star(A, B' .+ zeros(eltype(B),size(A)) ) # copy B so it adds to A row-wise.
    w' * (counts(sum(AB .⊻  R, dims = 2),0:n) - counts(sum(AB, dims = 2),0:n) )
end
