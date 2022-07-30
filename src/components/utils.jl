"""
    blockdiagonal(mats::Vector{Matrix{T}})

Constructs a block diagonal matrix from `mats`.
"""
function blockdiagonal(mats::Vector{Matrix{T}}) where T
    M, N = mapreduce(size, .+, mats)
    res = zeros(T, M, N)
    cur_ind = CartesianIndex(0, 0)
    for m in mats
        m_inds = CartesianIndices(m)
        for i in m_inds
            res[i + cur_ind] = m[i]
        end
        cur_ind += m_inds[end]
    end
    res
end