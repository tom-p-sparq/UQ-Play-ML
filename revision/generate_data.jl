module PlayMLGenerator

using DataFrames
using StatsBase
using Distributions
using CSV
using Random
using Sobol

export demand_curve, revenue_optimal_price, make_Thompson_Sampling


function demand_curve(x, X, E, D)
    p = 2*E
    return D / (1 + (x/X)^p)
end

function revenue_optimal_price(X, E)
    p = 2*E
    return X*(p-1)^(-1/p)
end

function prior_qmc_sample!(prior_sample; X = (30, 70), E = (1, 5), D = (1_000, 10_000))
    N = size(prior_sample, 1)
    ss = skip(SobolSeq([X[1], E[1], D[1]], [X[2], E[2], D[2]]), N)
    foreach(Base.Fix1(next!, ss), eachrow(prior_sample))
    return
end

function perturb_sample!(candidate_matrix, sample_matrix, incremental_log_likelihoods; X = (30, 70), E = (1, 5), D = (1_000, 10_000), conc=100)
    foreach(Iterators.enumerate(zip(eachrow(candidate_matrix), eachrow(sample_matrix)))) do (i, (c, s))
        Xs, Es, Ds = s
        # Normalise sample to (0, 1) - these are the beta modes
        xs = (Xs - X[1])/(X[2]-X[1])
        es = (Es - E[1])/(E[2]-E[1])
        ds = (Ds - D[1])/(D[2]-D[1])
        # beta distributions
        beta_xs = Beta(1 + conc*xs, 1 + conc*(1-xs))
        beta_es = Beta(1 + conc*es, 1 + conc*(1-es))
        beta_ds = Beta(1 + conc*ds, 1 + conc*(1-ds))
        # sample
        xc = rand(beta_xs)
        ec = rand(beta_es)
        dc = rand(beta_ds)
        # Reverse normalisation and store
        c[1] = X[1] + (X[2]-X[1])*xc
        c[2] = E[1] + (E[2]-E[1])*ec
        c[3] = D[1] + (D[2]-D[1])*dc
        # Reverse beta distributions
        beta_xc = Beta(1 + conc*xc, 1 + conc*(1-xc))
        beta_ec = Beta(1 + conc*ec, 1 + conc*(1-ec))
        beta_dc = Beta(1 + conc*dc, 1 + conc*(1-dc))

        ell_c_to_s = logpdf(beta_xc, xs) + logpdf(beta_ec, es) + logpdf(beta_dc, ds)
        ell_s_to_c = logpdf(beta_xs, xc) + logpdf(beta_es, ec) + logpdf(beta_ds, dc)
        incremental_log_likelihoods[i] = ell_c_to_s - ell_s_to_c
    end
    return
end

function log_likelihood_function(S, P, X, E, D)
    mu = demand_curve(P, X, E, D)
    return logpdf(Poisson(mu), S)
end

function make_Thompson_Sampling(
        N::Integer,
        true_params_1::NTuple{3, Real},
        true_params_2::NTuple{3, Real} = true_params_1;
        alpha=1.0,
        week=7,
        X = (30, 70),
        E = (1, 5),
        D = (1_000, 10_000)
    )
    
    df = DataFrame()
    df_increment = DataFrame()
    _P = zeros(Float64, 2*week)
    _S = zeros(Int64, 2*week)

    sample_matrix = zeros(N, 3)
    candidate_matrix = zeros(N, 3)
    incremental_log_likelihoods = zeros(N)
    log_likelihoods = zeros(N)

    prior_qmc_sample!(sample_matrix; X=X, E=E, D=D)

    for day in 1:(2*week)
    
        # sample
        i = sample(1:N)
        _X, _E, _D = sample_matrix[i, :]
        _P[day] = revenue_optimal_price(_X, _E)
    
        # observe
        parameters = (day <= week) ? true_params_1 : true_params_2
        _S[day] = rand(Poisson(demand_curve(_P[day], parameters...)))
    
        # record
        df_increment[!, :day] = fill(day, N)
        df_increment[!, :sample] = collect(1:N)
        df_increment[!, :X] = sample_matrix[:, 1]
        df_increment[!, :E] = sample_matrix[:, 2]
        df_increment[!, :D] = sample_matrix[:, 3]
        df_increment[!, :LLH] = log_likelihoods
        df_increment[!, :price_opt] = revenue_optimal_price.(sample_matrix[:, 1], sample_matrix[:, 2])
        df_increment[!, :demand_opt] = demand_curve.(df_increment[!, :price_opt], eachcol(sample_matrix)...)
        df_increment[!, :rev_opt] = df_increment[!, :price_opt] .* df_increment[!, :demand_opt]
        df_increment[!, :TS] = map(==(i), 1:N)
        df_increment[!, :P_TS] = fill(_P[day], N)
        df_increment[!, :S_TS] = fill(_S[day], N)
        append!(df, df_increment)

        # update
        f = row -> sum(1:day) do d
            (alpha ^ (day-d)) * log_likelihood_function(_S[d], _P[d], row...)
        end

        for K in 1:100
            perturb_sample!(candidate_matrix, sample_matrix, incremental_log_likelihoods)
            for i in 1:N
                s = selectdim(sample_matrix, 1, i)
                c = selectdim(candidate_matrix, 1, i)
                inc_log_lh = incremental_log_likelihoods[i]
                if (f(c) - f(s) + inc_log_lh > 0) || (f(c) - f(s) + inc_log_lh > log(rand()))
                    s .= c
                end
            end
        end
        map!(f, log_likelihoods, eachrow(sample_matrix))
    end
    print("Week 1: True parameters $true_params_1 give optimal price $(revenue_optimal_price(true_params_1[1], true_params_1[2]))")
    print("Week 2: True parameters $true_params_2 give optimal price $(revenue_optimal_price(true_params_2[1], true_params_2[2]))")
    return df
end

function make_Thompson_Sampling(path::String, args...; kwargs...)
    CSV.write(path, make_Thompson_Sampling(args...; kwargs...))
    return true
end



end