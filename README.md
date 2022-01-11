# Pauli error estimation via population recovery

This is an implementation of the algorithm in S. T. Flammia and R. O'Donnell, "Pauli error estimation via population recovery", *Quantum* **5**, 549 (2021); arXiv:2105.02885. 


```julia
# load the functions
include("PauliEstViaPopRec.jl")
```

We consider an example with $n=5$ qubits and $k=25$ nonzero Pauli error probabilities sampled at random. With $m=10^6$ samples, we don't expect to be able to reconstruct error rates with probabilities less than about $10^{-3}$. 


```julia
n = 5 # number of qubits
m = 10^6 # number of measurements
k = 25 # size of channel support
δ = 0.2 # probability of some nontrivial error occurring

# Define a Pauli channel as a tuple (p,P), 
# where p is a list of probabilities and P are Pauli row vectors (0,1,2,3)
(p,P) = RandomPauliChannel(n,k,δ)

# get m random probe states
A = rand(Int8.(1:3),m,n)

# Measure the channel (p,P) using the probe states A
R = Measure(A,p,P)

# convert the tuple (p,P) into a dictionary for ease of comparison
pP = sparseP(p,P)
```


```julia
Dict{Vector{Int8}, Float64} with 25 entries:
  [2, 0, 1, 3, 2] => 0.00286122
  [1, 2, 0, 1, 1] => 0.00262801
  [1, 2, 3, 3, 0] => 0.00971039
  [1, 2, 1, 2, 3] => 0.006714
  [3, 0, 1, 3, 1] => 0.00774545
  [1, 3, 0, 0, 2] => 0.00695082
  [2, 0, 0, 1, 1] => 0.0118502
  [1, 2, 2, 1, 3] => 0.0236053
  [1, 2, 1, 1, 2] => 0.0176598
  [0, 1, 3, 1, 1] => 0.00252022
  [1, 1, 3, 3, 3] => 0.0441254
  [3, 1, 2, 2, 3] => 0.00394266
  [0, 3, 2, 2, 0] => 0.00661846
  [0, 3, 3, 3, 3] => 0.00310002
  [0, 0, 0, 3, 2] => 0.00800969
  [2, 0, 3, 0, 2] => 0.00426104
  [3, 3, 2, 3, 2] => 0.01067
  [0, 0, 0, 0, 0] => 0.8
  [2, 1, 0, 3, 3] => 0.00629306
  [3, 2, 3, 2, 2] => 0.00806051
  [1, 2, 1, 1, 0] => 0.00391613
  [3, 1, 1, 1, 0] => 0.00288139
  [1, 3, 3, 2, 0] => 0.00245237
  [2, 1, 3, 3, 2] => 0.00104182
  [2, 2, 3, 0, 0] => 0.00238201
```

We can run the algorithm with a specific choice of threshold value for pruning. Here we just choose $1/\sqrt{m}$, even though the rigorous theorem requires using some log factors. We get a pretty accurate reconstruction, as quantified by the TVD.


```julia
ϵ = 1/sqrt(m) # pick a simple choice for the thresholding value
@time (q,Q) = EstChan(A,R,ϵ)
qQ = sparseP(q,Q)
```

```julia
  6.518071 seconds (335.72 k allocations: 7.448 GiB, 11.23% gc time, 1.21% compilation time)
```

```julia
Dict{Vector{Int8}, Float64} with 24 entries:
  [2, 0, 1, 3, 2] => 0.00264581
  [1, 2, 0, 1, 1] => 0.00246216
  [1, 2, 3, 3, 0] => 0.0090255
  [1, 2, 1, 2, 3] => 0.006663
  [3, 0, 1, 3, 1] => 0.00769397
  [1, 3, 0, 0, 2] => 0.00687487
  [2, 0, 0, 1, 1] => 0.0116704
  [1, 2, 2, 1, 3] => 0.0235204
  [1, 2, 1, 1, 2] => 0.0175356
  [0, 1, 3, 1, 1] => 0.00266447
  [1, 1, 3, 3, 3] => 0.0439482
  [3, 1, 2, 2, 3] => 0.00390891
  [0, 3, 2, 2, 0] => 0.00675103
  [0, 3, 3, 3, 3] => 0.00309328
  [0, 0, 0, 3, 2] => 0.00789797
  [2, 0, 3, 0, 2] => 0.0043065
  [3, 2, 3, 2, 2] => 0.00822019
  [0, 0, 0, 0, 0] => 0.800279
  [2, 1, 0, 3, 3] => 0.00632972
  [3, 3, 2, 3, 2] => 0.010396
  [1, 2, 1, 1, 0] => 0.00376725
  [3, 1, 1, 1, 0] => 0.00290316
  [1, 3, 3, 2, 0] => 0.00213225
  [2, 2, 3, 0, 0] => 0.00242184
```

Here is the total variation distance:


```julia
tvd(pP,qQ)
```


```julia
0.0023034701460466077
```

The error bars are fairly tight as well.


```julia
σ = EstChanErrorBars(A,Q,R)
mean(σ)
```


```julia
0.00020968924879989553
```

What about the estimates that we threw away with our choice of threshold? We can see what those estimates look like by computing all the probability estimates for every Pauli string, even the ones less than $\delta$ but that are still positive. What does the TVD look like now? It is much worse, but this is not surprising because we are overfitting.


```julia
@time (s,S) = EstAll(A,R)
sS = sparseP(s,S)
tvd(pP,sS)
```

```julia
 31.902549 seconds (786.36 k allocations: 34.350 GiB, 9.75% gc time, 0.50% compilation time)
```

```julia
0.041446673271046584
```

We can do a scatter plot to compare the complete list of nonnegative estimates to the thresholded values and the "true" values. 


```julia
using Plots
```


```julia
# convert the dictionary key quaternary digits to plotable numbers
keys2num(kK,n) = hcat(collect(keys(kK))...)' * (4 .^((n-1):-1:0))
vals2num(kK) = collect(values(kK))

scatter(keys2num(pP,n), vals2num(pP), 
    yaxis = ("log10(p)",:log10, [1e-4,1e-1]),
    xaxis = "Pauli number",
    label = "p (true)", markershape = :x, markersize = 4)
scatter!(keys2num(qQ,n), vals2num(qQ),
    label = "q (estimate)", markershape = :hline, markersize = 6,
    markerstrokecolor = :auto, yerror = σ)
scatter!(keys2num(sS,n), vals2num(sS),
    label = "all estimates", markershape = :+, markersize = 3, markeralpha = .3)
```

![svg](fig.svg)

