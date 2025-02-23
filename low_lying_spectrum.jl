using ITensors, ITensorMPS



include("./utils.jl")


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



function generate_dephaser(sites, δt, κ)
    single_site_deph = exp(-2 * κ * δt * (Id_Id - Z_Z))
    rho = MPO(length(sites))
    for i = 1:length(sites)
        newA = op(single_site_deph, sites, i)
        rho[i] = newA
    end
    return rho
end

function generate_U(sites, δt, J)


    two_site_operator_forward = exp(-im * J * kron(Z_forward, Z_forward) * δt)
    two_site_operator_backward = exp(im * J * kron(Z_backward, Z_backward) * δt)
    two_site_operator = two_site_operator_forward * two_site_operator_backward


    rho = MPO(length(sites))

    operator = op(two_site_operator, sites, 1, 2)
    U, S, V = svd(operator, (sites[1], sites[1]'))
    rho[1] = U * S
    rho[2] = V

    @show inds(rho[1]), inds(rho[2])

    for i = 2:length(sites)-1
        operator = op(two_site_operator, sites, i, i + 1)
        U, S, V = svd(operator, (sites[i], sites[i]'))

        L = U * S
        R = V

        replaceind!(L, inds(L)[1], prime(inds(L)[1]))
        replaceind!(L, inds(L)[2], prime(inds(L)[2]))

        rho[i] = L * (rho[i])
        rho[i+1] = R


    end

    ITensors.replaceprime!(rho, 2, 1)
    truncate!(rho, cutoff = 1e-10)


    return rho


end

function generate_kick(sites, δt, ϵ)
    single_site_kic = exp(-im * ϵ * δt * X_forward) * exp(im * ϵ * δt * X_backward)

    rho = MPO(length(sites))
    for i = 1:length(sites)
        newA = op(single_site_kic, sites, i)
        rho[i] = newA
    end
    return rho

end


let
    ITensors.set_warn_order(20)
    N = 5
    d = 2
    sites = siteinds(d^2, N)
    κ = 0.5
    δt = 0.1
    J = 3
    ϵ = 1

    V_t_avg = generate_dephaser(sites, δt, κ)

    @show V_t_avg

    U_tilde = generate_U(sites, δt, J)

    @show U_tilde

    K = generate_kick(sites, δt, ϵ)


    U = apply(U_tilde, V_t_avg)

    T = apply(K, U)

    T_dag = dagger_MPO(T)

    superHamiltonian = ITensors.replaceprime!(T_dag * T, 2, 1)

    psi0 = create_Neel(sites)
    identity = create_identity(sites, d)

    nsweeps = 400
    maxdim = [10, 10, 10, 10, 10]
    cutoff = [1E-10]
    energy, gs = dmrg([superHamiltonian], psi0; nsweeps, maxdim, cutoff)


    @show inner(identity, gs)

    states = [gs]

    for i = 1:12

        println(i)

        psi0 = create_Neel(sites)

        nsweeps = 150
        maxdim = [40]
        cutoff = [1E-10]
        noise = [1E-8]
        weight = 2500

        energy, psi = dmrg(
            superHamiltonian,
            states,
            psi0;
            nsweeps,
            maxdim,
            cutoff,
            weight,
            noise,
            ishermitian = true,
        )


        push!(states, psi)

    end

end
