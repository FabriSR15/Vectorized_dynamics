using ITensors, ITensorMPS

include("./utils.jl")


S_z = [1/2 0; 0 -1/2]
S_x = [0 1/2; 1/2 0]
S_y = [0 -im*1/2; im*1/2 0]

S_plus = (S_x + im * S_y)
S_minus = (S_x - im * S_y)
Id = [1 0; 0 1]


#In analogy with Keldish, I'll name the operators "forward" when acting to the right and "backward" when acting to the left. 
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
    g = 1
    gamma = 2

    sites = siteinds(d^2, N)

    #Here, we reproduce the results of https://arxiv.org/pdf/1501.06786

    #We start by giving the Liouvillian an MPO form 
    os = OpSum()
    for j = 1:N
        os += -im * g, S_z_forward, j
        os -= -im * g, S_z_backward, j
    end

    for j = 1:N-1
        os += gamma^2, S_minus_forward, j, conj(S_minus_backward), j
        os += gamma^2, S_minus_forward, j + 1, conj(S_minus_backward), j + 1
        os += gamma^2, S_minus_forward, j + 1, conj(S_minus_backward), j
        os += gamma^2, S_minus_forward, j + 1, conj(S_minus_backward), j + 1
    end

    for j = 1:N-1
        os += -1 / 2 * gamma^2, S_plus_forward, j, S_minus_forward, j
        os += -1 / 2 * gamma^2, S_plus_forward, j, S_minus_forward, j + 1
        os += -1 / 2 * gamma^2, S_plus_forward, j + 1, S_minus_forward, j
        os += -1 / 2 * gamma^2, S_plus_forward, j + 1, S_minus_forward, j + 1
    end


    for j = 1:N-1
        os += -1 / 2 * gamma^2, transpose(S_minus_backward), j, conj(S_minus_backward), j
        os +=
            -1 / 2 * gamma^2, transpose(S_minus_backward), j, conj(S_minus_backward), j + 1
        os +=
            -1 / 2 * gamma^2, transpose(S_minus_backward), j + 1, conj(S_minus_backward), j
        os += -1 / 2 * gamma^2,
        transpose(S_minus_backward),
        j + 1,
        conj(S_minus_backward),
        j + 1
    end

    H = MPO(os, sites)


    #Currently we need to create the dag(H) by hand, what is happening with dag(H) in ITensor??

    os2 = OpSum()
    for j = 1:N
        os2 += im * g, S_z_forward, j
        os2 -= im * g, S_z_backward, j
    end

    for j = 1:N-1
        os2 += gamma^2, S_minus_forward', j, conj(S_minus_backward)', j
        os2 += gamma^2, S_minus_forward', j + 1, conj(S_minus_backward)', j + 1
        os2 += gamma^2, S_minus_forward', j + 1, conj(S_minus_backward)', j
        os2 += gamma^2, S_minus_forward', j + 1, conj(S_minus_backward)', j + 1
    end


    for j = 1:N-1
        os2 += -1 / 2 * gamma^2, S_minus_forward', j, S_plus_forward', j
        os2 += -1 / 2 * gamma^2, S_plus_forward', j, S_minus_forward', j + 1
        os2 += -1 / 2 * gamma^2, S_plus_forward', j + 1, S_minus_forward', j
        os2 += -1 / 2 * gamma^2, S_minus_forward', j + 1, S_plus_forward', j + 1
    end


    for j = 1:N-1
        os2 += -1 / 2 * gamma^2, conj(S_minus_backward)', j, transpose(S_minus_backward)', j
        os2 += -1 / 2 * gamma^2,
        transpose(S_minus_backward)',
        j,
        conj(S_minus_backward)',
        j + 1
        os2 += -1 / 2 * gamma^2,
        transpose(S_minus_backward)',
        j + 1,
        conj(S_minus_backward)',
        j
        os2 += -1 / 2 * gamma^2,
        conj(S_minus_backward)',
        j + 1,
        transpose(S_minus_backward)',
        j + 1
    end


    H = MPO(os, sites)
    H_dag = MPO(os2, sites)

    H_3 = apply(H_dag, H)

    #Finally, we create the observable to monitor, which in the case of the paper is S^2_y
    os3 = OpSum()
    for j = 1:N
        os3 += 1 / N, S_y_forward, j
    end
    S_y_op = MPO(os3, sites)
    S_y_squared_op = apply(S_y_op, S_y_op)





    #Now we can set the DMRG calculation
    psi0 = create_Neel(sites)
    identity = create_identity(sites, d)

    H_pen = -0.005 * outer(identity', identity)



    nsweeps = 400
    maxdim = [10, 10, 10, 10, 10]
    cutoff = [1E-10]
    energy, psi = dmrg([H_3, H_pen], psi0; nsweeps, maxdim, cutoff, ishermitian = true)

    maxdim = 15
    energy, psi = dmrg([H_3, H_pen], psi; nsweeps, maxdim, cutoff, ishermitian = true)






    @show inner(identity, apply(S_y_squared_op, psi))
    @show inner(identity, psi)
    @show inner(identity, apply(S_y_squared_op, psi)) / inner(identity, psi)


    return




end
