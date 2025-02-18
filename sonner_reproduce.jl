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


function create_dephaser_boundary(κ, δt, i, l)

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    #@show single_site_deph
    dephaser = diag_itensor(ComplexF64, i', i, l)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 

    return dephaser

end 

function create_zoz(J, δt, l, j)

    M = zeros(ComplexF64, 4, 4)

    #Configurations with same spin in both branches (diagonal)
    for x = 1:4 
        M[x, x] = 1/2
    end 

    M[1, 2] = exp(-2 * im * J * δt)
    M[1, 3] = exp(2 * im * J * δt)
    M[1, 4] = 1

    M[2, 3] = 1
    M[2, 4] = exp(2 * im * J * δt)
    M[3, 4] = exp(-2 * im * J * δt)

    M = M + transpose(M)

    zoz = ITensor(M, l, j)

    return zoz

end 

function create_kick(ϵ, δt, i)

    M = exp(-im * ϵ * δt * X_forward) * exp(im * ϵ * δt * X_backward)

    kick = ITensor(M, i', i)

    return kick 

end 

#I'm assuming that the initial state is \rho_0 is the inf temp 
function create_first_operator_boundary(κ, J, ϵ, δt, site, link)

    dummy_1 = Index(4, "dummy1")

    #REMEMBER TO CHANGE TO 
    #A = [1 0 0 1]
    A = 1/2 * [1 0 0 1]
    initial_tensor = ITensor(A, dummy_1)

    dummy_2 = Index(4, "dummy2")

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    #@show single_site_deph
    dephaser = diag_itensor(ComplexF64, dummy_1', dummy_1, dummy_2)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 

    zoz = create_zoz(J, δt, dummy_2, site)

    temp = zoz * (dephaser * initial_tensor)
    
    noprime!(temp)

    kick = create_kick(ϵ, δt, dummy_1)

    temp = kick * temp 

    #@show delta(link, dummy_1')

    return delta(link, dummy_1') * temp 

end 

function create_bulk_operator_boundary(κ, J, ϵ, δt, site, link1, link2)

    dummy_1 = Index(4, "dummy1")
    dummy_2 = Index(4, "dummy2")
    

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    #@show single_site_deph
    dephaser = diag_itensor(ComplexF64, dummy_1, link1, dummy_2)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 

    zoz = create_zoz(J, δt, dummy_2, site)
    temp = zoz * dephaser
    
    kick = create_kick(ϵ,δt, dummy_1)

    temp = kick * temp 

    return delta(link2, dummy_1') * temp 


end 

function create_last_operator_boundary(κ, J, ϵ, δt, site, link1)

    dummy_1 = Index(4, "dummy1")
    dummy_2 = Index(4, "dummy2")
    

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, dummy_1, link1, dummy_2)
    for x = 1:4 
        dephaser[x, x, x] = single_site_deph[x, x] 
    end 

    zoz = create_zoz(J, δt, dummy_2, site)
    temp = zoz * dephaser
    
    kick = create_kick(ϵ, δt,dummy_1)

    temp = kick * temp 

    A = [1 0 0 1]
    partial_trace = ITensor(A, dummy_1')

    temp = partial_trace * temp

    return temp 


end 

function create_initial_mps(κ, J, ϵ, δt, sites)
    psi = random_mps(sites, linkdims = 4)
    N = length(psi)

    #First 
    psi[1] = create_last_operator_boundary(κ, J, ϵ, δt, sites[1], inds(psi[1])[2])
    #bulk 
    for n = 2:N-1
        psi[n] = create_bulk_operator_boundary(κ, J, ϵ, δt, sites[n], inds(psi[n])[3], inds(psi[n])[1])
    end 

    #final (or initial in time, as one wants to see it)
    psi[length(psi)] = create_first_operator_boundary(κ, J, ϵ, δt, sites[length(psi)], inds(psi[N])[1])

    return psi 
end 


function create_first_operator_bulk(κ, J, ϵ, δt, site, link)

    dummy_1 = Index(4, "dummy1")


    #REMEMBER TO CHANGE
    #A = [1 0 0 1]

    A = 1/2 * [1 0 0 1]
    initial_tensor = ITensor(A, dummy_1)

    dummy_2 = Index(4, "dummy2")

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    #@show single_site_deph
    dephaser = diag_itensor(ComplexF64, dummy_1', dummy_1, dummy_2, site)
    for x = 1:4 
        dephaser[x, x, x, x] = single_site_deph[x, x] 
    end 

    zoz = create_zoz(J, δt, dummy_2, site')

    temp = zoz * (dephaser * initial_tensor)

    kick = create_kick(ϵ, δt,dummy_1')

    temp = kick * temp 

    #@show delta(link, dummy_1'')

    temp = delta(link, dummy_1'') * temp

    return ITensors.permute(temp, inds(temp)[2], inds(temp)[3], inds(temp)[1])


end 

function create_bulk_operator_bulk(κ, J, ϵ, δt, site, link1, link2)
    dummy_1 = Index(4, "dummy1")
    dummy_2 = Index(4, "dummy2")
    

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, dummy_1, link1, dummy_2, site)
    for x = 1:4 
        dephaser[x, x, x, x] = single_site_deph[x, x] 
    end 

    zoz = create_zoz(J, δt, dummy_2, site')
    temp = zoz * dephaser
    
    kick = create_kick(ϵ, δt, dummy_1)

    temp = kick * temp 

    temp = delta(link2, dummy_1') * temp

    return ITensors.permute(temp, inds(temp)[2], inds(temp)[4], inds(temp)[3], inds(temp)[1])


end 


function create_last_operator_bulk(κ, J, ϵ, δt, site, link1)
    dummy_1 = Index(4, "dummy1")
    dummy_2 = Index(4, "dummy2")
    

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, dummy_1, link1, dummy_2, site)
    for x = 1:4 
        dephaser[x, x, x, x] = single_site_deph[x, x] 
    end 

    zoz = create_zoz(J, δt, dummy_2, site')
    temp = zoz * dephaser
    
    kick = create_kick(ϵ, δt, dummy_1)

    temp = kick * temp 

    A = [1 0 0 1]
    partial_trace = ITensor(A, dummy_1')

    temp = partial_trace * temp

    return  ITensors.permute(temp, inds(temp)[1], inds(temp)[3], inds(temp)[2])


end 


function create_mpo(κ, J, ϵ, δt, sites)
    ρ = random_mps(sites; linkdims=2)
    ρ = outer(ρ', ρ)
    N = length(ρ)

    #First 
    ρ[1] = create_last_operator_bulk(κ, J, ϵ, δt, sites[1], inds(ρ[1])[3])
    #bulk 
    for n = 2:N-1
        ρ[n] = create_bulk_operator_bulk(κ, J, ϵ, δt, sites[n], inds(ρ[n])[3], inds(ρ[n])[4])
    end 

    #final (or initial in time, as one wants to see it)
    ρ[N] = create_first_operator_bulk(κ, J, ϵ, δt, sites[N], inds(ρ[N])[3])

    return ρ
end 




#=let 
    J = pi/4 - 0.06
    δt = 1
    κ = 0.3
    ϵ = pi/4 - 0.06
    d = 2

    SvN_array = []
    N_array = []

    for N = 2:2:30
        sites = siteinds(d^2, N)
        println(N)


        #test everything 
        




        #@show psi 
        #@show inds(create_first_operator_boundary(κ, J, ϵ, δt, sites[10], inds(psi[10])[1]))
        #@show inds(create_bulk_operator_boundary(κ, J, ϵ, δt, sites[9], inds(psi[9])[3], inds(psi[9])[1]))

        #=@show create_initial_mps(κ, J, ϵ, δt, sites)

        @show create_first_operator_bulk(κ, J, ϵ, δt, sites[10], inds(psi[10])[1])

        @show create_first_operator_bulk(κ, J, ϵ, δt, sites[10], inds(psi[10])[1])=#

        infl = create_initial_mps(κ, J, ϵ, δt, sites)

        @show norm(infl)

        #infl /= norm(infl)
        mpo = create_mpo(κ, J, ϵ, δt, sites)

        @show inner(infl, infl)

        #@show von_Neumann_entropy(infl, Int(N/2))
        for x = 1:100
            infl = apply(mpo, infl; maxdim=64)
            @show inner(infl, infl)
            #infl /= norm(infl) 
            #println(N, " ", von_Neumann_entropy(infl, Int(N/2)))
        end 

        #push!(N_array, N)
        #push!(SvN_array, von_Neumann_entropy(infl, Int(N/2)))

    #@show create_last_operator_bulk(κ, J, ϵ, δt, sites[1], inds(psi[1])[2])
    end 

    #@show N_array
    #@show SvN_array
end
=#
