using ITensors, ITensorMPS


X = [0 1; 1 0]

function create_diamond_boundary(h, i, l)

    diamond = diag_itensor(ComplexF64, i', i, l)
    diamond[1, 1, 1] = exp(im * h)
    diamond[2, 2, 2] = exp(-im*h)

    return diamond

end 

function create_diamond_bulk(h, i, ll, lr)

    diamond = diag_itensor(ComplexF64, i', i, ll, lr)

    diamond[1, 1, 1, 1] = exp(im * h)
    diamond[2, 2, 2, 2] = exp(-im*h)

    return diamond


end 

function create_zoz(J, l)
    M = [exp(-im * J) exp(im * J); exp(im * J) exp(-im * J)]

    j = Index(2, "j")

    zoz = ITensor(M, l, j)

    return zoz


end 

function create_kick(ϵ, i)

    M = exp(-im * ϵ * X)

    kick = ITensor(M, i', i)

    return kick 

end 

#This creates the boundary dual transfer matrix
#for τ steps  
function create_boundary_dual_transfer(site, h, J, ϵ, τ)

    link = Index(2, "l1")
    diamond = create_diamond_boundary(h, site, link)
    zoz = create_zoz(J, link)
    kick = create_kick(ϵ, site)

    T = prime(kick) * (diamond * zoz)
    ITensors.replaceprime!(T, 2, 1)

    for N = 2:τ
        @show inds(T)

        link = Index(2, "l" * string(N))
        diamond = create_diamond_boundary(h, site, link)
        zoz = create_zoz(J, link)
        kick = create_kick(ϵ, site)
        newT = prime(kick) * (diamond * zoz)
        ITensors.replaceprime!(newT, 2, 1)
        T = prime(newT)*T
        ITensors.replaceprime!(T, 1, 0)
        ITensors.replaceprime!(T, 2, 1)
    end 

    return T 

end 

function create_bulk_dual_transfer(T_boundary, site, sites_num, h, J, ϵ, τ)


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


end 




let 
    J = 2
    h = 3
    ϵ = 4
    τ = 20
    sites = siteinds(2, 3)

    link = Index(2, "l")
    diamond = create_diamond_boundary(h, sites[1], link)
    zoz = create_zoz(J, link)
    kick = create_kick(ϵ, sites[1])

    
    result = create_boundary_dual_transfer(sites[1], h, J, ϵ, τ)
    
    #@show inds(result)

    #new_res = create_bulk_dual_transfer(result, sites[2], 2,  h, J, ϵ, τ)

    #new_new_res = create_bulk_dual_transfer(new_res, sites[3], 3,  h, J, ϵ, τ)

end 