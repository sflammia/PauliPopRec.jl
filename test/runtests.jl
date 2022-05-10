using Test
using PauliPopRec

# Test membership in the probability simplex
issimplex(p) = sum(p) ≈ 1 && all( p .≥ 0)

@testset "PauliPopRec utils tests" begin

    ### Testing randp

    # the point p should lie in the simplex
    p = PauliPopRec.randp(3)
    @test all(p .≥ 0) && sum(p) ≈ 1

    # first element of p should be 1 minus the second argument
    p = PauliPopRec.randp(3,0.1)
    @test all(p .≥ 0) && sum(p) ≈ 1 && p[1] == 0.9


    ### Testing samplepaulichannel
    
    # Test that a trivial channel returns the same samples
    P = PauliPopRec.PauliChannel( Int8[0, 0] => 1.0)
    s = PauliPopRec.samplepaulichannel(P,3)
    @test s == Int8[0 0; 0 0; 0 0]
    
    # Test that a 50/50 channel returns fair samples
    P = PauliPopRec.PauliChannel( Int8[0] => 0.5, Int8[1] => 0.5)
    s = PauliPopRec.samplepaulichannel(P,100000)
    @test isapprox( sum(s)/length(s),  0.5, atol = 1e-2 )


    ### Testing star

    # This tests all possible single-qubit commutator relations.
    A = Int8.(kron([0 1 2 3],[1, 1, 1, 1])); 
    B = Int8.(kron([1 1 1 1],[0, 1, 2, 3]));
    C = Int8[0  0  0  0;
             0  0  1  1; 
             0  1  0  1; 
             0  1  1  0];
    @test PauliPopRec.star(A,B) == C

    # Test the broadcast for different sized inputs
    # A row vector should broadcast down the rows
    A = [1 2; 2 3; 3 1]
    B = [1 3]
    C = Int8[0 1; 1 0; 1 1]
    @test PauliPopRec.star(A,B) == C

    # A column vector should broadcast across the columns 
    B = [1, 2, 3]
    C = Int8[0 1; 0 1; 0 1]
    @test PauliPopRec.star(A,B) == C

    # Test the three-argument version of star
    Ax, Az = PauliPopRec.bits(A)
    B = [1, 3]
    C = Int8[0 1; 1 0; 1 1]
    # First test this decomposition of bits
    @test A == eltype(A)(2) .* Az .+ Ax
    # Now test correctness
    @test PauliPopRec.star(B,Ax,Az) == C

    ### Testing individual_recovery

    # Test a few different results with simple answers.
    n = 2; m = 3;
    B = zeros(Int8,2)
    A = ones(Int8,m,n)
    Ax = Int8.(rem.(A,2))
    Az = A .>> 1
    R = [0 0; 0 1; 1 0] # hamming weights are 0, 1, 1.
    w = (-1/2).^(0:2) 
    est = PauliPopRec.individual_recovery(B,Ax,Az,R,w,n)
    # answer should be 1, since 1*(-1/2)^0 + 2*(-1/2)^1 = 0
    @test est ≈ 0

    B = ones(Int8,2)
    est = PauliPopRec.individual_recovery(B,Ax,Az,R,w,n)
    # answer should be -3 in this example
    @test est ≈ -3

    B *= 2
    est = PauliPopRec.individual_recovery(B,Ax,Az,R,w,n)
    # answer should be -1.5 in this example
    @test est ≈ -1.5

end  # utils testset


@testset "PauliPopRec main tests" begin

    ### Testing randpaulichannel

    # Test some properties of random Pauli channels
    n = 3
    k = 4
    P = randpaulichannel(n,k)
    p = collect(values(P))
    E = collect(keys(P))
    @test issimplex(p)
    @test length(p) == k
    @test size(hcat(E...)) == (n,k)

    n = 5
    k = 20
    P = randpaulichannel(n,k,0.1)
    p = collect(values(P))
    E = collect(keys(P))
    @test issimplex(p)
    @test length(p) == k
    @test size(hcat(E...)) == (n,k)


    ### Testing probepaulichannel

    # probing a bit-flip channel with X eigenstates
    n = 2
    m = 100
    A = ones(Int8,m,n)
    P = PauliPopRec.PauliChannel( Int8[0] => 0.5, Int8[1] => 0.5)
    R = probepaulichannel(P, A)
    @test sum(R) == 0

    # now send X eigenstates through a Y or Z channel
    P = PauliPopRec.PauliChannel( Int8[2] => 0.5, Int8[3] => 0.5)
    R = probepaulichannel(P, A)
    @test sum(R) == m*n


    ### Testing pauli_indrec

    # If A == R == ones(Int8,m,n), then our estimate of the identity is 1/4
    B = Int8[0,0]
    @test pauli_indrec(B,A,R) == 0.25
    
    B = Int8[1,1]
    @test pauli_indrec(B,A,R) == -0.75


    ### Testing pauli_poprec

    # Test that the algorithm works and doesn't return total nonsense.
    n = 5 # number of qubits
    m = 10^4 # number of measurements
    A = rand(Int8.(1:3), m, n);
    k = 25 # size of channel support
    δ = 0.2 # probability of some nontrivial error occurring
    P = randpaulichannel(n, k, δ)
    R = probepaulichannel(P, A);
    ϵ = 1/sqrt(m) # pick a simple choice for the thresholding value
    Q = pauli_poprec(A, R, ϵ)
    @test pauli_tvd(P,Q) < 0.5
    
    
    ### Testing pauli_stderr
    # Test the previous estimates 
    err = pauli_stderr(Q, A, R)
    @test all(0.0 .<= err .< 0.1)


    ### Testing pauli_totalrecall

    # Test that the estimates are positive
    QQ = pauli_totalrecall(A,R)
    @test all( 1.0 .>= collect(values(QQ)) .>= 0.0 )

    
    ### Testing pauli_tvd

    # tvd should be 0.0 with itself
    p = [0.9, 0.1]; 
    PE = Vector{Int8}[[0], [1]]; 
    P = PauliPopRec.PauliChannel( PE .=> p )
    @test pauli_tvd(P,P) ≈ 0.0
    
    # tvd should be 0.1 since nontrivial supports are disjoint
    q = [0.9, 0.1]; 
    QE = Vector{Int8}[[0], [2]]; 
    Q = PauliPopRec.PauliChannel( QE .=> q )
    @test pauli_tvd(P,Q) ≈ 0.1    

end  # utils testset
