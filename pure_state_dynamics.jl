using ITensors, ITensorMPS
using Plots



Z = [1 0; 0 -1]
X = [0 1; 1 0]
Y = [0 -im; im 0]
Id = [1 0; 0 1]
Id_Id = kron(Id, Id)
Z_Z = kron(Z, Z)

Z_forward = kron(Z, Id)
Z_backward = kron(Id, Z)

X_forward = kron(X, Id)
X_backward = kron(Id, X)

σ_plus = 1 / 2 * (X + im * Y)
σ_minus = 1 / 2 * (X - im * Y)



let
    ITensors.set_warn_order(20)
    N = 10
    d = 2
    sites = siteinds("S=1/2", N)
    κ = 0.3
    δt = 0.01
    T = 0.2
    J = pi / 4 - 0.06
    ϵ = pi / 4 - 0.06


    state = [isodd(n) ? "Up" : "Up" for n = 1:N]
    psi0 = MPS(sites, state)

    gates = ITensor[]

    for j = 1:N
        s = sites[j]
        hj = 2 * κ * op("Sz", s)
        Gj = exp(-im * δt * hj)
        push!(gates, Gj)
    end

    trotter_gates = ITensor[]
    for j = 1:(N-1)
        s1 = sites[j]
        s2 = sites[j+1]
        hj = 4 * J * op("Sz", s1) * op("Sz", s2)
        Gj = exp(-im * δt / 2 * hj)
        push!(trotter_gates, Gj)
    end
    append!(trotter_gates, reverse(trotter_gates))
    append!(gates, trotter_gates)

    for j = 1:N
        s = sites[j]
        hj = 2 * ϵ * op("Sx", s)
        Gj = exp(-im * δt * hj)
        push!(gates, Gj)
    end

    Sz_array = []
    t_array = []

    psi = deepcopy(psi0)
    cutoff = 1e-8

    for t = 0.0:δt:T
        Sz = expect(psi, "Sz"; sites = 10)

        push!(Sz_array, 2 * Sz)
        push!(t_array, t)
        println("$t $Sz")
        t ≈ T && break
        psi = apply(gates, psi; cutoff)
        normalize!(psi)

        @show maxlinkdim(psi)
    end
    plot(t_array, Sz_array)



end
