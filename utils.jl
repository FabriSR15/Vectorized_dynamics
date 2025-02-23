using ITensors, ITensorMPS


Z = [1 0; 0 -1]
X = [0 1; 1 0]
Y = [0 -im; im 0]
H = 1 / sqrt(2) * [1 1; 1 -1]
Id = [1 0; 0 1]
Id_Id = kron(Id, Id)
Z_Z = kron(Z, Z)
H_H = kron(H, H)


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

function create_Neel(sites)
    psi = randomMPS(sites; linkdims = 1)

    N = length(sites)

    idx1, idx2 = inds(psi[1])
    O1 = onehot(idx1 => 1, idx2 => 1)

    psi[1] = O1

    idx1, idx2 = inds(psi[N])
    O1 = onehot(idx1 => 1, idx2 => 4)

    psi[N] = O1

    for i = 2:N-1
        idx1, idx2, idx3 = inds(psi[i])
        if i % 2 == 0
            O1 = onehot(idx1 => 1, idx2 => 4, idx3 => 1)
        else
            O1 = onehot(idx1 => 1, idx2 => 1, idx3 => 1)
        end

        psi[i] = O1
    end

    return psi
end

function create_ferromagnetic(sites)

    psi = randomMPS(sites; linkdims = 1)

    N = length(sites)

    idx1, idx2 = inds(psi[1])
    O1 = onehot(idx1 => 1, idx2 => 1)

    psi[1] = O1

    idx1, idx2 = inds(psi[N])
    O1 = onehot(idx1 => 1, idx2 => 1)

    psi[N] = O1

    for i = 2:N-1
        idx1, idx2, idx3 = inds(psi[i])

        O1 = onehot(idx1 => 1, idx2 => 1, idx3 => 1)

        psi[i] = O1
    end

    return psi
end

function create_identity(sites, d)

    id = randomMPS(sites; linkdims = 1)
    N = length(sites)

    for i = 1:N
        for j = 1:d^2
            if j == 1 || j == 4
                id[i][j] = 1
            else
                id[i][j] = 0
            end
        end
    end

    return id

end


function create_Hadamard(sites)

    psi = create_ferromagnetic(sites)
    for i = 1:length(psi)
        H_op = op(H_H, sites, i)
        psi[i] = noprime!(H_op * psi[i])
    end

    return psi

end

function dagger_MPO(ρ::ITensorMPS.MPO)

    conj_rho = conj(ρ)
    ITensors.replaceprime!(conj_rho, 0, 2)

    return conj_rho

end

function von_Neumann_entropy(psi::MPS, c::Int)
    s = siteinds(psi)
    orthogonalize!(psi, c)
    N = length(psi)

    if N == 2
        _, S = svd(psi[c], (s[c]))
    else
        _, S = svd(psi[c], (linkind(psi, c - 1), s[c]))
    end


    SvN = 0.0
    for n = 1:dim(S, 1)
        p = S[n, n]^2
        SvN -= p * log(p)
    end

    return SvN
end
