using ITensors, ITensorMPS


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


sites = siteinds(4, 3)

J = 1.1234
δt = 3.8
two_site_operator_forward = exp(-im * J * kron(Z_forward, Z_forward) * δt)
two_site_operator_backward = exp(im * J * kron(Z_backward, Z_backward) * δt)
two_site_operator = two_site_operator_forward * two_site_operator_backward
@show two_site_operator

#Now, instead of the diamond, we create a dephaser due to averaging
function create_dephaser_boundary(κ, δt, i, l)

    single_site_deph = exp(-2 * κ * δt * (Id_Id - Z_Z))
    @show single_site_deph
    dephaser = diag_itensor(ComplexF64, i', i, l)
    for x = 1:4
        dephaser[x, x, x] = single_site_deph[x, x]
    end

    return dephaser

end


##We should create for 0 zoz, simply by making the 4x4 matrix, which is pretty easy to construct
##The dumb way of creating the MPO is quite terrible 

function create_zoz(J, δt, l)

    M = zeros(ComplexF64, 4, 4)

    #Configurations with same spin in both branches (diagonal)
    for x = 1:4
        M[x, x] = exp(-2 * im * J * δt) / 2
    end

    M[1, 2] = 1
    M[1, 3] = 1
    M[1, 4] = exp(2 * im * J * δt)

    M[2, 3] = exp(2 * im * J * δt)
    M[2, 4] = 1
    M[3, 4] = 1

    M = M + transpose(M)

    j = Index(4, "j")

    zoz = ITensor(M, l, j)

    return zoz

end

function create_kick(ϵ, i)

    M = exp(-im * ϵ * X_forward) * exp(im * ϵ * X_backward)

    kick = ITensor(M, i', i)

    return kick

end

#This creates the boundary dual transfer matrix
#for τ steps  
function create_boundary_dual_transfer(site, κ, J, δt, ϵ, τ)

    link = Index(4, "l1")
    diamond = create_dephaser_boundary(κ, δt, site, link)
    zoz = create_zoz(J, δt, link)
    kick = create_kick(ϵ, site)

    T = prime(kick) * (diamond * zoz)
    ITensors.replaceprime!(T, 2, 1)

    for N = 2:τ
        println(N)

        link = Index(4, "l" * string(N))
        diamond = create_dephaser_boundary(κ, δt, site, link)
        zoz = create_zoz(J, δt, link)
        kick = create_kick(ϵ, site)
        newT = prime(kick) * (diamond * zoz)
        ITensors.replaceprime!(newT, 2, 1)
        T = prime(newT) * T
        ITensors.replaceprime!(T, 1, 0)
        ITensors.replaceprime!(T, 2, 1)
    end

    return T

end

#=function create_bulk_dual_transfer(T_boundary, site, sites_num, h, J, ϵ, τ)


    ll = inds(T_boundary)[end]
    lr = Index(2, "lr1")
    diamond = create_diamond_bulk(h, site, ll, lr)
    zoz = create_zoz(J, lr)
    kick = create_kick(ϵ, site)

    T = prime(kick) * (diamond * zoz)
    ITensors.replaceprime!(T, 2, 1)
    result = T_boundary * T

    println("This")

    for N = 2*(sites_num-1): (2*(sites_num-1) + (τ-2))

        ll = inds(T_boundary)[end-N]
        println(N, ll)
        lr = Index(2, "lr" * string(N))
        diamond = create_diamond_bulk(h, site, ll, lr)
        zoz = create_zoz(J, lr)
        kick = create_kick(ϵ, site)
        newT = prime(kick) * (diamond * zoz)
        ITensors.replaceprime!(newT, 0, 1; tags="Site,n="*string(sites_num))
        result = newT*result
        ITensors.replaceprime!(result, 2, 1)
    end 

    return result 


end=#




let
    J = 2
    κ = 1
    ϵ = 4
    δt = 0.1
    τ = 18
    sites = siteinds(4, 3)

    link = Index(4, "l")


    result = create_boundary_dual_transfer(sites[1], κ, J, δt, ϵ, τ)
    @show result

    #@show inds(result)

    #new_res = create_bulk_dual_transfer(result, sites[2], 2,  h, J, ϵ, τ)

    #new_new_res = create_bulk_dual_transfer(new_res, sites[3], 3,  h, J, ϵ, τ)

end
