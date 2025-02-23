using ITensors, ITensorMPS
using Plots


function swapcols!(X::AbstractMatrix, i::Integer, j::Integer)
    @inbounds for k = 1:size(X, 1)
        X[k, i], X[k, j] = X[k, j], X[k, i]
    end
end

function swaprows!(X::AbstractMatrix, i::Integer, j::Integer)
    @inbounds for k = 1:size(X, 2)
        X[i, k], X[j, k] = X[j, k], X[i, k]
    end
end

S_z = [1/2 0; 0 -1/2]
S_x = [0 1/2; 1/2 0]
S_y = [0 -im*1/2; im*1/2 0]

S_plus = S_x + im * S_y
S_minus = S_x - im * S_y
Id = [1 0; 0 1]


let

    N = 10
    d = 2
    t = 0.1
    s = siteinds(d^2, N)

    println(length(s))

    #Creation of the initial density matrix 
    #= chi = 1

     psi = randomMPS(s;linkdims=chi)

     idx1, idx2 = inds(psi[1])
     O1 = onehot(idx1 => 1, idx2 => 1)

     psi[1] = O1 

     idx1, idx2 = inds(psi[N])
     O1 = onehot(idx1 => 1, idx2 => 4)

     psi[N] = O1 

     for i = 2:N-1
         idx1, idx2, idx3 = inds(psi[i])
         if i % 2 == 0
             O1 = onehot(idx1 =>1, idx2 => 4, idx3 => 1)
         else 
             O1 = onehot(idx1 =>1, idx2 => 1, idx3 => 1)
         end 

         psi[i] = O1 
     end 

     println(psi[3])

     #Creation of the (vectorized) identity operator 
     chi = 1

     id = randomMPS(s;linkdims=chi)

     for i = 1:N 
         for j = 1:d^2
             if j == 1 || j == 4 
                 id[i][j] = 1 
             else 
                 id[i][j] = 0
             end 
         end 
     end 

     S_x_full = kron(S_x, S_x)
     Id_full = kron(Id, Id)
     A = kron(S_x_full, Id_full)
     @show A

     for i=0:d-1, j = 0:d-1
         for k = 0:d-2, l = k:d-1
             a = d^3*i + d^2 * k + d*l + j + 1
             b = d^3*i + d^2 * l + d*k + j + 1
             swapcols!(A, a, b)
             swaprows!(A, a, b)
         end 
     end 
     @show A


     #
     h_j = kron(S_z, S_z) + 1/2 * kron(S_plus, S_minus) + 1/2 * kron(S_minus, S_plus)

     h_j = exp(-im * t/2 * h_j)

     U = kron(h_j, conj(h_j))

     for i=0:d-1, j = 0:d-1
         for k = 0:d-2, l = k:d-1
             a = d^3*i + d^2 * k + d*l + j + 1
             b = d^3*i + d^2 * l + d*k + j + 1
             swapcols!(U, a, b)
             swaprows!(U, a, b)
         end 
     end 

     gates = ITensor[]
     for j = 1:(N-1)
         s1 = s[j]
         s2 = s[j+1]
         G_j = op(U, [s[j], s[j+1]])
         push!(gates, G_j)
     end 

     append!(gates, reverse(gates))

     ttotal = 5

     Obs = kron(S_z, Id)
     O = op(Obs, s[5])

     println(O)

     t_array = []
     S_z_array = []
     for tau in 0.0:0.1:ttotal

         psi_copy = deepcopy(psi)
         new_A = O * psi_copy[5]
         psi_copy[5] = new_A
         noprime!(psi[5])

         push!(t_array, tau)
         push!(S_z_array, real(inner(id, psi_copy)))

         println(tau)
         tau≈ttotal && break

         psi = apply(gates, psi)

         #println(inner(id, psi))
     end

     plot(t_array, S_z_array) =#
end
