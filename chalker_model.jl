#This script has several utilities related to computing the influence functional (IF) of the model
#shown in PUT Chalker paper 

using ITensors, ITensorMPS, Plots, LinearAlgebra



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



SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]


#We will use U and V 
#SWAP has the nice property that after reordering for MPO form it
#keeps its form 


include("./utils.jl")


#Once again the we use the disordered longitudinal field 


function u_SWAP(J, δt)
    temp = exp(im * J * δt * SWAP)
    result = copy(temp)

    result[1, 3] = temp[2, 1]
    result[1, 4] = temp[2, 2]
    result[2, 1] = temp[1, 3]
    result[2, 2] = temp[1, 4]

    result[3, 3] = temp[4, 1]
    result[3, 4] = temp[4, 2]
    result[4, 1] = temp[3, 3]
    result[4, 2] = temp[3, 4]


    #@show temp

    #@show result




    return result

end

function U_and_V(J, δt)
    matrix = u_SWAP(J, δt)
    F = svd(matrix)

    U = F.U * Diagonal(F.S)

    #=@show F.S 

    @show F.U * Diagonal(F.S) * F.Vt

    @show size(F.Vt)=#

    U = reshape(U, (2, 2, 4))

    V = reshape(F.Vt, (4, 2, 2))
    return U, V
end


function create_dephaser_boundary(κ, δt, i)

    single_site_deph = exp(-2 * κ * δt * (Id_Id - Z_Z))
    #@show single_site_deph
    dephaser = ITensor(single_site_deph, i', i)

    return dephaser

end


function create_first_operator_boundary(κ, δt, l, U, link)


    dummy_2 = Index(4, "dummy2")
    A = [1 0 0 1]
    initial_tensor = 1 / 2 * ITensor(A, dummy_2)

    deph = create_dephaser_boundary(κ, δt, dummy_2)

    U = reshape(U, (4, 4))

    U_full = kron(U, conj(U))

    U_full = reshape(U_full, (4, 4, 16))

    U_tensor = ITensor(U_full, link, dummy_2', l)

    single_site_op = U_tensor * (deph * initial_tensor)

    ITensors.replaceprime!(single_site_op, 2, 1)

    return single_site_op

end


function create_bulk_operator_boundary(κ, δt, l, U, link1, link2)

    deph = create_dephaser_boundary(κ, δt, link2)

    U = reshape(U, (4, 4))

    U_full = kron(U, conj(U))

    U_full = reshape(U_full, (4, 4, 16))

    U_tensor = ITensor(U_full, link1, link2', l)

    single_site_op = U_tensor * deph

    return single_site_op

end




let
    J = pi / 4
    δt = 1
    κ = 0.00
    ϵ = pi / 4
    d = 2
    N = 10
    sites = siteinds(16, N)

    U = U_and_V(J, δt)[1]

    ψ = random_mps(sites; linkdims = 4)

    @show create_bulk_operator_boundary(κ, δt, sites[5], U, inds(ψ[5])[1], inds(ψ[5])[3])
end
