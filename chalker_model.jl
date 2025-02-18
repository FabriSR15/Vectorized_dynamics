#This script has several utilities related to computing the influence functional (IF) of the model
#shown in PUT Chalker paper 

using ITensors, ITensorMPS, Plots 



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



SWAP = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]


#We will use U and V 
#SWAP has the nice property that after reordering for MPO form it
#keeps its form 


include("./utils.jl")


#Once again the we use the disordered longitudinal field 


function u_SWAP(J, δt)
    temp = exp(i * J * δt * SWAP)
    u_SWAP = copy(temp)

    u_SWAP[1, 3] = temp[2, 1]
    u_SWAP[1, 4] = temp[2, 2]
    u_SWAP[2, 1] = temp[1, 3]
    u_SWAP[2, 2] = temp[1, 4]

    u_SWAP[3, 3] = temp[4, 1]
    u_SWAP[3, 4] = temp[4, 2]
    u_SWAP[4, 1] = temp[3, 3]
    u_SWAP[4, 2] = temp[3, 4]

    return u_SWAP

end

function U_and_V(J, δt)
    U, S, V = svd(u_SWAP(J, δt))
    return U, S, V
end 


function create_dephaser_boundary(κ, δt, i)

    single_site_deph = exp(-2 * κ * δt * (Id_Id - Z_Z))
    #@show single_site_deph
    dephaser = ITensor(single_site_deph, i', i)

    return dephaser 

end 


@show U_and_V(5, 0.1)



