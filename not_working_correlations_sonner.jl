using ITensors, ITensorMPS


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


include("./utils.jl")
include("./sonner_reproduce.jl")



function create_im(κ, J, ϵ, δt, sites)

    infl = create_initial_mps(κ, J, ϵ, δt, sites)

    #infl /= norm(infl)
    mpo = create_mpo(κ, J, ϵ, δt, sites)

    #@show von_Neumann_entropy(infl, Int(N/2))
    for x = 1:150
        infl = apply(mpo, infl; maxdim=128)
        #@show inner(infl, infl)
        #infl /= norm(infl) 
        #println(N, " ", von_Neumann_entropy(infl, Int(N/2)))
    end 

    return infl

end 


function create_first_operator_edge(κ, ϵ, δt, site, link)
    dummy_1 = Index(4, "dummy1")

    A = Z_forward * transpose([1 0 0 1])
    initial_tensor = 1/2 * ITensor(A, dummy_1)

    #O_z = ITensor(Z_forward, dummy_1, new_dummy)

    #initial_tensor = O_z * initial_tensor

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    #@show single_site_deph
    dephaser = diag_itensor(ComplexF64, dummy_1', dummy_1, site)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 

    temp = dephaser * initial_tensor

    kick = create_kick(ϵ, dummy_1')

    temp = kick * temp 

    #@show delta(link, dummy_1'')

    temp = delta(link, dummy_1'') * temp

    return temp 


end 

function create_last_operator_edge(κ, ϵ, δt, site, link1)
    dummy_1 = Index(4, "dummy1")
    

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, dummy_1, link1, site)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 
    
    kick = create_kick(ϵ, dummy_1)

    temp = kick * dephaser 

    A = [1 0 0 1]
    A = A * Z_forward
    partial_trace = ITensor(A, dummy_1')

    temp = partial_trace * temp

    return ITensors.permute(temp, inds(temp)[2], inds(temp)[1])
    
end

function create_bulk_operator_edge(κ, ϵ, δt, site, link1, link2)
    dummy_1 = Index(4, "dummy1")
    

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, dummy_1, link1, site)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 
    
    kick = create_kick(ϵ, dummy_1)

    temp = kick * dephaser 

    temp = delta(link2, dummy_1') * temp 
    return ITensors.permute(temp, inds(temp)[1], inds(temp)[3], inds(temp)[2])

end 


function create_central_site(κ, ϵ, δt, sites)
    ρ = random_mps(sites; linkdims=4)
    N = length(ρ)

    #First 
    ρ[1] = create_last_operator_edge(κ, ϵ, δt, sites[1], inds(ρ[1])[2])
    #bulk 
    for n = 2:N-1
        ρ[n] = create_bulk_operator_edge(κ, ϵ, δt, sites[n], inds(ρ[n])[3], inds(ρ[n])[1])
    end 

    #final (or initial in time, as one wants to see it)
    ρ[N] = create_first_operator_edge(κ, ϵ, δt, sites[N], inds(ρ[N])[1])

    return ρ
end 


let 
    J = pi/4 - 0.06
    δt = 1
    κ = 0.88
    ϵ = pi/4 - 0.06
    d = 2


    N = 6

    sites = siteinds(d^2, N)
    psi = random_mps(sites, linkdims = 4)

    @show inds(psi)

    #first_op = create_first_operator_edge(κ, ϵ, δt, sites[N], inds(psi[N])[1])
    #last_op = create_last_operator_edge(κ, ϵ, δt, sites[1], inds(psi[1])[2])
    #bulk_op = create_bulk_operator_edge(κ, ϵ, δt, sites[3], inds(psi[3])[3], inds(psi[3])[1])



    infl = create_im(κ, J, ϵ, δt, sites)
    final_spin = create_central_site(κ, ϵ, δt, sites)

    
    @show inner(conj(infl), final_spin)

end 