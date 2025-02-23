using ITensors, ITensorMPS, Plots


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


include("./utils.jl")
include("./sonner_reproduce.jl")



function create_im(κ, J, ϵ, δt, sites)

    N = length(sites)

    infl = create_initial_mps(κ, J, ϵ, δt, sites)

    mpo = create_mpo(κ, J, ϵ, δt, sites)

    final_mpo = create_mpo(κ, J - 0.06, ϵ, δt, sites)

    #@show von_Neumann_entropy(infl, Int(N/2))
    for x = 1:20
        infl = apply(mpo, infl; maxdim = 256, cutoff = 1e-10)
        @show maxlinkdim(infl)
        #println(N, " ", von_Neumann_entropy(infl, Int(N/2)))
    end

    infl = apply(final_mpo, infl; maxdim = 256)

    return infl #von_Neumann_entropy(infl, Int(N/2))

end


function create_first_operator_edge(κ, ϵ, δt, site, link)
    dummy_1 = Index(4, "dummy1")
    #new_dummy = Index(4, "new_dummy")



    #REMEMBER TO CHANGE
    A = 1 / 2 * [1 0 0 1]
    initial_tensor = ITensor(Z_forward * transpose(A), dummy_1)

    #O_z = ITensor(Z_forward, dummy_1, new_dummy)

    #initial_tensor = O_z * initial_tensor

    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    #@show single_site_deph
    dephaser = diag_itensor(ComplexF64, site, dummy_1', dummy_1)
    for x = 1:4
        dephaser[x, x, x] = single_site_deph[x, x]
    end

    temp = dephaser * initial_tensor

    kick = create_kick(ϵ, δt, dummy_1')

    temp = kick * temp

    #@show delta(link, dummy_1'')

    temp = delta(link, dummy_1'') * temp

    return temp


end


function create_bulk_operator_edge(κ, ϵ, δt, site, link1, link2)
    dummy_1 = Index(4, "dummy1")


    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, site, dummy_1, link1)
    for x = 1:4
        dephaser[x, x, x] = single_site_deph[x, x]
    end

    kick = create_kick(ϵ, δt, dummy_1)

    temp = kick * dephaser

    temp = delta(link2, dummy_1') * temp
    return temp


end

function create_last_operator_edge(κ, ϵ, δt, site, link1)

    dummy_1 = Index(4, "dummy1")


    single_site_deph = exp(-im * κ * Z_forward * δt) * exp(im * κ * Z_backward * δt)
    dephaser = diag_itensor(ComplexF64, site, dummy_1, link1)
    for x = 1:4
        dephaser[x, x, x] = single_site_deph[x, x]
    end

    kick = create_kick(ϵ, δt, dummy_1)

    temp = kick * dephaser

    A = [1 0 0 1]

    partial_trace = ITensor(A * Z_backward, dummy_1')

    temp = partial_trace * temp

    return temp


end



function create_edge(κ, ϵ, δt, sites)
    ρ = random_mps(sites; linkdims = 4)
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

    J = pi / 4
    δt = 1
    κ = 0.29
    ϵ = pi / 4
    d = 2


    contraction_array = []
    time_array = []
    for N = 2:20
        println(N)
        t = N * δt
        sites = siteinds(d^2, N)


        infl = create_im(κ, J, ϵ, δt, sites)

        @show maxlinkdim(infl)

        edge = create_edge(κ, ϵ - 0.06, δt, sites)


        contraction = infl[1] * edge[1]


        for x = 2:N
            contraction = contraction * infl[x]
            contraction = contraction * edge[x]
        end

        push!(contraction_array, real(contraction[1]))
        push!(time_array, t)

    end

    @show contraction_array
    #plot(time_array, contraction_array, yaxis = :log)

    #=random_psi = random_mps(sites; linkdims = 4)

    @show inds(random_psi)

    first_site_edge = create_first_operator_edge(κ, ϵ, δt, sites[N], inds(random_psi[N])[1])
    #bulk_site_edge = create_bulk_operator_edge(κ, ϵ, δt, sites[3], inds(random_psi[3])[3], inds(random_psi[3])[1])

    #@show inds(infl)
    #@show first_site_edge
    #@show bulk_site_edge
    #@show edge

    contraction = infl[1] * edge[1]

    for x in 2:N
        contraction = contraction * infl[x]
        contraction = contraction * edge[x]
        @show contraction
    end =#


    #println([1 0 0 1] * Z_forward)
    #@show infl[6] * first_site_edge


    #=δϵ = LinRange(0.0, 0.28, 50)

    delta_S = []
    delta_eps = []
    for x in δϵ
        J = pi/4 - x 
        ϵ = pi/4 - x
        sites8 = siteinds(d^2, 8)
        S8 = create_im(κ, J, ϵ, δt, sites8)[2]
        sites10 = siteinds(d^2, 10)
        S10 = create_im(κ, J, ϵ, δt, sites10)[2]

        push!(delta_eps, x)
        push!(delta_S, (S10-S8)/2)

    end 
    @show delta_S
    @show delta_eps


    plot(delta_eps, delta_S)=#

    #=for N in 2:2:28
        sites = siteinds(d^2, N)
        infl = create_im(κ, J, ϵ, δt, sites)
    end =#




end
