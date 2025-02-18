using ITensors, ITensorMPS
using Plots 

include("./utils.jl")


S_z = 2* [1/2 0; 0 -1/2]
S_x = 2* [0 1/2; 1/2 0]
S_y = 2* [0 -im * 1/2; im * 1/2 0]

S_plus = 1/2 * (S_x + im * S_y)
S_minus = 1/2 * (S_x - im * S_y)
Id = [1 0; 0 1]

S_z_forward = kron(S_z, Id)
S_x_forward = kron(S_x, Id)
S_y_forward = kron(S_y, Id)
S_plus_forward = kron(S_plus, Id)
S_minus_forward = kron(S_minus, Id)

S_z_backward = kron(Id, S_z)
S_x_backward = kron(Id, S_x)
S_y_backward = kron(Id, S_y)
S_plus_backward = kron(Id, S_plus)
S_minus_backward = kron(Id, S_minus)


let 
    N = 10
    d = 2
    g = 4
    V = 2
    gamma = 1

    sites = siteinds(d^2, N)

    g_gamma_array = []
    avg_sx_array = []
    avg_sy_array = []
    avg_sz_array = []

    for g = 0.1:0.1:4
        println(g)
        #Here, we reproduce the results of https://arxiv.org/pdf/1501.06786

        #We start by giving the Liouvillian an MPO form 
        os = OpSum()
        for j = 1:N-1
            os += -im * V/4, S_z_forward, j, S_z_forward, j+1 
            os -= -im * V/4, transpose(S_z_backward), j, transpose(S_z_backward), j+1
        end 

        for j = 1:N
            os += -im * g/2, S_x_forward, j
            os -= -im * g/2, S_x_backward, j
        end 


        for j = 1:N
            os += gamma^2, S_minus_forward * conj(S_minus_backward), j
        end 

        for j = 1:N
            os += -1/2 * gamma^2, S_plus_forward * S_minus_forward, j
        end 


        for j = 1:N
            os += -1/2 * gamma^2, transpose(S_minus_backward) * conj(S_minus_backward), j
        end 

        H = MPO(os, sites)


        #Currently we need to create the dag(H) by hand, what is happening with dag(H) in ITensor??

        os2 = OpSum()
        for j = 1:N-1
            os2 += im * V/4, S_z_forward, j, S_z_forward, j+1 
            os2 -= im * V/4, transpose(S_z_backward), j, transpose(S_z_backward), j+1
        end 

        for j = 1:N
            os2 += im * g/2, S_x_forward, j
            os2 -= im * g/2, S_x_backward, j
        end 

        for j = 1:N
            os2 += gamma^2, (S_minus_forward * conj(S_minus_backward))', j
        end 

        for j = 1:N
            os2 += -1/2 * gamma^2, (S_plus_forward * S_minus_forward)', j
        end 


        for j = 1:N
            os2 += -1/2 * gamma^2, (transpose(S_minus_backward) * conj(S_minus_backward))', j
        end 


        H = MPO(os, sites)
        H_dag = MPO(os2, sites)

        H_3 = apply(H_dag, H)

        #Finally, we create the observable to monitor, which in the case of the paper is S^2_y
        os3 = OpSum()
        for j = 1:N
            os3 += 1/N, S_x_forward, j
        end 
        S_x_op = MPO(os3, sites)

        os4 = OpSum()
        for j = 1:N
            os4 += 1/N, S_y_forward, j
        end 
        S_y_op = MPO(os4, sites)

        os5 = OpSum()
        for j = 1:N
            os5 += 1/N, S_z_forward, j
        end 
        S_z_op = MPO(os5, sites)



        #Now we can set the DMRG calculation
        psi0 = create_Neel(sites)
        identity = create_identity(sites, d)


        
        nsweeps = 200
        maxdim = [10, 10, 10, 10, 10] 
        cutoff = [1E-10]
        energy,psi = dmrg([H_3],psi0;nsweeps,maxdim,cutoff, ishermitian=true)

        maxdim = 15
        energy,psi = dmrg([H_3],psi;nsweeps,maxdim,cutoff, ishermitian=true)






        @show inner(identity, apply(S_x_op, psi))
        @show inner(identity, psi)
        @show real(inner(identity, apply(S_x_op, psi)))/real(inner(identity, psi))

        push!(g_gamma_array, g/gamma)
        push!(avg_sx_array, real(inner(identity, apply(S_x_op, psi)))/real(inner(identity, psi)))
        push!(avg_sy_array, real(inner(identity, apply(S_y_op, psi)))/real(inner(identity, psi)))
        push!(avg_sz_array, real(inner(identity, apply(S_z_op, psi)))/real(inner(identity, psi)))
    end 

    println(g_gamma_array)
    println(avg_sx_array)

    scatter(g_gamma_array, avg_sx_array)
    savefig("firsttrialsx.png")
    scatter(g_gamma_array, avg_sy_array)
    savefig("firsttrialsy.png")
    scatter(g_gamma_array, avg_sz_array)
    savefig("firsttrialsz.png")

    return




end 