## -------------  Model

@with_kw mutable struct Model
    # Number of endowments
    K :: Int64 = 5
    # Number of households
    N :: Int64 = 10
    # Number of options
    M :: Int64 = 12
    # How often to simulate each type
    nSim :: Int64 = 200

    # Governs how deviation function shows intermediate results
    showFreq :: Int64 = 100
    currentIter :: Int64 = 0
end


# No of calibrated params
function n_params(mdl :: Model)
    return 4 * mdl.K
end



## ---------  Model parameters

mutable struct ModelParams
    alphaV :: Vector{Float64}
    betaV :: Vector{Float64}
    gammaV :: Vector{Float64}
    deltaV :: Vector{Float64}
end

function show_params(mp :: ModelParams)
    print("Alpha:  "); println(round.(mp.alphaV, digits = 3));
    print("Beta:   "); println(round.(mp.betaV, digits = 3));
    print("Gamma:  "); println(round.(mp.gammaV, digits = 3));
    print("Delta:  "); println(round.(mp.deltaV, digits = 3));
end


## --------  Model stats

mutable struct ModelStats
    # Fraction of choices in each [choice class, endowment class, by endowment] bin
    frac_cekM  ::  Array{Float64, 3}
end

function n_tg_moments(ms :: ModelStats)
    return length(ms.frac_cekM)
end


# ------------------