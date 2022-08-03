function kalman_predict(x, P, H, F, Q, R)
    x = F*x
    P = F*P*F' + Q
    y = H*x
    S = H*P*H' + R
    x, P, y, S
end

function kalman_update(x, P, r, S, H, R)
    K = P*H'/S
    x = x + K*r
    P = (I - K*H)*P*(I - K*H)' + K*R*K'
    S = H*P*H' + R
    y = H*x
    x, P, y, S
end

@model function kalman(sts, ys; forecast=0)
    obs_dim, N = size(ys)
    latent_dim = latent_size(sts)
 
    x₀ ~ MvNormal(zeros(latent_dim), I)
    ϵ ~ filldist(truncated(Normal(0, 1); lower=0), latent_dim)
    σ ~ filldist(truncated(Normal(0, 1); lower=0), obs_dim)
    
    P = diagm(ϵ.^2)
    R = diagm(σ.^2)
    M = N + forecast
    x = x₀
    res = map(1:M) do t
        H, F, Q = sts(t)
        x, P, y, S = kalman_predict(x, P, H, F, Q, R)

        if t <= N
            r = ys[:,t] - y
            x, P, y, S = kalman_update(x, P, r, S, H, R)
            Turing.@addlogprob! - 0.5 * sum(logdet(S) + r'inv(S)*r)
        end
        (;x, P, y, S)
    end
 
    return (
        x=[r.x for r in res],
        Σx=[r.P for r in res],    
        y=[r.y for r in res],
        Σy=[r.S for r in res]
    )
end

kalman(sts; kwargs...) = kalman(sts, zeros(1, 0); kwargs...)