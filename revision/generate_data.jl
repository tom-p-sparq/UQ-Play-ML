module PlayMLGenerator

using DataFrames
using StatsBase
using Distributions
using CSV
using Random

export demand_curve, revenue_optimal_price, make_demand_curves, make_revenue_optimal_prices, make_Thompson_Sampling

const xrange = 2:2:100
const Xrange = 30:1:70
const Erange = 1.0:0.2:5.0
const Drange = 1_000:1_000:10_000
const rng = Xoshiro(123)

function demand_curve(x, X, E, D)
    p = 2*E
    return D / (1 + (x/X)^p)
end

function revenue_optimal_price(X, E)
    p = 2*E
    return X*(p-1)^(-1/p)
end

function make_demand_curves()
    r = Iterators.product(xrange, Xrange, Erange, Drange)
    df = DataFrame(r, [:price, :X, :E, :D])
    df.demand = demand_curve.(df.price, df.X, df.E, df.D)
    df.revenue = df.price .* df.demand
    return df
end

function make_revenue_optimal_prices()
    r = Iterators.product(Xrange, Erange)
    df = DataFrame(r, [:X, :E])
    df.opt_price = revenue_optimal_price.(df.X, df.E)
    return df
end

function make_Thompson_Sampling(X1, E1, D1, X2=X1, E2=E1, D2=D1; alpha=0.0, week=7)
    r = Iterators.product(Xrange, Erange, Drange)
    N = length(r)
    df = DataFrame(r, [:X, :E, :D])
    logw = zeros(N)

    for days in 1:(2*week)
        # sample
        theta = sample(rng, eachrow(df), Weights(exp.(logw)))
        P = revenue_optimal_price(theta.X, theta.E)
        # observe
        params = if days <= week 
            (X1, E1, D1)
        else
            (X2, E2, D2)
        end
        S = rand(rng, Poisson(demand_curve(P, params...)))
        # forget
        logw .*= (1.0 - alpha)
        # update
        @. logw += logpdf(Poisson(demand_curve(P, df.X, df.E, df.D)), S)
        logw .-= maximum(logw)

        df[!, Symbol("price$days")] = fill(P, N)
        df[!, Symbol("sales$days")] = fill(S, N)
        df[!, Symbol("w$days")] = exp.(logw)
    end

    df_opt = make_revenue_optimal_prices()
    return leftjoin(df, df_opt, on=[:X, :E])
end

function make_demand_curves(path)
    CSV.write(path, make_demand_curves())
    return true
end

function make_revenue_optimal_prices(path)
    CSV.write(path, make_revenue_optimal_prices())
    return true
end

function make_Thompson_Sampling(path::String, args...; kwargs...)
    CSV.write(path, make_Thompson_Sampling(args...; kwargs...))
    return true
end



end