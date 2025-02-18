using ITensors, ITensorMPS
using Plots 



include("./utils.jl")


Z = [1 0; 0 -1]
X = [0 1; 1 0]
Y = [0 -im; im  0]
Id = [1 0; 0 1]
Id_Id = kron(Id, Id)
Z_Z = kron(Z, Z)

Z_forward = kron(Z, Id)
Z_backward = kron(Id, Z)

X_forward = kron(X, Id)
X_backward = kron(Id, X)
 
σ_plus = 1/2 * (X + im * Y)
σ_minus = 1/2 * (X - im * Y)



function generate_dephaser(sites, δt, κ)
    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    rho = MPO(length(sites))
    for i = 1:length(sites)
        newA = op(single_site_deph, sites, i)
        rho[i] = newA
    end 
    return rho 
end 

function generate_U(sites, δt, J)
    two_site_operator_forward = exp(- im * J * kron(Z_forward, Z_forward) * δt)
    two_site_operator_backward =  exp(im * J * kron(Z_backward, Z_backward) * δt)
    two_site_operator = two_site_operator_forward * two_site_operator_backward


    rho = MPO(length(sites))

    operator = op(two_site_operator, sites, 1, 2)
    U, S, V = svd(operator, (sites[1], sites[1]'))
    rho[1] = U*S
    rho[2] = V

    @show inds(rho[1]), inds(rho[2])

    for i = 2:length(sites)-1
        operator = op(two_site_operator, sites, i, i+1)
        U, S, V = svd(operator, (sites[i], sites[i]'))

        L = U*S
        R = V

        replaceind!(L, inds(L)[1], prime(inds(L)[1]))
        replaceind!(L, inds(L)[2], prime(inds(L)[2]))

        #@show inds(L)
        #@show inds(V) 

        rho[i] = L * (rho[i]) 
        rho[i+1] = R

        #@show rho 
    

    end 

    ITensors.replaceprime!(rho, 2, 1)


    
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
    N = 100 
    d = 2
    sites = siteinds(d^2, N)
    κ = 0.3
    δt = 1
    J = pi/4 - 0.06
    ϵ = pi/4 - 0.06

    V_t = generate_dephaser(sites, δt, κ)

    U_tilde = generate_U(sites, δt, J)

    U = apply(U_tilde, V_t)

    K = generate_kick(sites, δt, ϵ)

    M = apply(K, U)


    psi0 = create_ferromagnetic(sites)
    identity = create_identity(sites, d)

    os5 = OpSum()
    os5 += Z_forward, 50
    Z_op = MPO(os5, sites)

    @show inner(identity, apply(Z_op, psi0))

    Z_array = [real(inner(identity, apply(Z_op, psi0)))/real(inner(identity, psi0))]

    psi = deepcopy(psi0) 
    χ_max = []
    for x = 1:500
        println(x)
        psi = apply(M, psi; cutoff=1e-10) 
        println(maxlinkdim(psi))
        push!(χ_max, maxlinkdim(psi))
        @show real(inner(identity, apply(Z_op, psi)))/real(inner(identity, psi))
        push!(Z_array, real(inner(identity, apply(Z_op, psi)))/real(inner(identity, psi)))

    end 

    plot(Z_array)
    #plot!(χ_max)
    

end 
