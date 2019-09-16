## -------------  Model

# Make random model parameters for testing and intialization
# All in [-1, 1]
# Changes GLOBAL_RNG
function make_random_parameters(mdl :: Model, randSeed :: Integer)
    Random.seed!(randSeed);
    return ModelParams(rand(mdl.K) .- 0.5, rand(mdl.K) .- 0.5, 
        rand(mdl.K) .- 0.5, rand(mdl.K) .- 0.5);
end


# Objective function to be minimized
function objective(mdl :: Model,  tgStats :: ModelStats,  mp :: ModelParams)
    mStats = solve(mdl, mp);

    # Deviation
    dev = deviation(mdl, tgStats, mStats);
    return dev
end


# Guess to model params and reverse
function guess_to_mp(mdl :: Model, guessV :: Vector{Float64})
    @assert length(guessV) == mdl.K * 4
    return ModelParams(guessV[1 : mdl.K],  guessV[(mdl.K + 1) : (2 * mdl.K)],
        guessV[(2 * mdl.K + 1) : (3 * mdl.K)],  guessV[(3 * mdl.K + 1) : (4 * mdl.K)])
end


function mp_to_guess(mdl :: Model, mp :: ModelParams)
    return vcat(mp.alphaV, mp.betaV, mp.gammaV, mp.deltaV)
end


## -----------  Solve

function solve(mdl :: Model, mp :: ModelParams)
    endow_kjM = make_endowments(mdl,  mdl.N,  mp.alphaV, mp.betaV);
    optValue_kmM = make_endowments(mdl, mdl.M,  mp.gammaV, mp.deltaV);

    prob_mjM = zeros(mdl.M, mdl.N);
    for j = 1 : mdl.N
        prob_mjM[:, j] = decision_prob(mdl, endow_kjM[:, j],  optValue_kmM);
    end

    choice_sjM = simulate_choices(mdl, prob_mjM, 320);

    mStats = compute_stats(mdl, choice_sjM, endow_kjM);

    return mStats
end


## Make endowments of households or options
#=
IN
    nInd
        Number of individuals for which endowments are drawn
=#
function make_endowments(mdl :: Model,  nInd :: Integer,
    alphaV :: Vector{Float64}, betaV :: Vector{Float64})

    endow_kjM = zeros(mdl.K, nInd);
    endowV = (collect(1 : nInd) ./ nInd) .^ 0.7;
    for k = 1 : mdl.K
        endow_kjM[k,:] = alphaV[k] .* (endowV .+ betaV[k] .* 10.0 .* endowV);
    end

    @assert all(abs.(endow_kjM) .> 0.0)
    return endow_kjM
end


## Decision probabilties for one household
# Inputs are endowments and option characteristics
# Must be nonlinear. Otherwise some parameters do not affect decision probs (intercepts)
function decision_prob(mdl :: Model, endow_kV :: Vector{Float64}, optValue_kmM :: Matrix{Float64})
    # Utility of alternatives
    utilV = zeros(mdl.M);
    for m = 1 : mdl.M
        # Ensures positive utility
        utilV[m] = exp(sum(endow_kV .* optValue_kmM[:, m]));
    end
    probV = utilV ./ sum(utilV);
    return probV
end


# This changes GLOBAL_RNG
function simulate_choices(mdl :: Model, prob_mjM, randSeed :: Integer)
    Random.seed!(randSeed);
    choice_sjM = zeros(Int64, mdl.nSim, mdl.N);
    for j = 1 : mdl.N
        d = Categorical(prob_mjM[:, j]);
        choice_sjM[:, j] = rand(d, mdl.nSim);
    end
    return choice_sjM
end


"""
	compute_stats

Compute model statistics
"""
function compute_stats(mdl :: Model,  choice_sjM :: Matrix{T1}, 
    endow_kjM :: Matrix{Float64}) where T1 <: Integer

    choiceCl_mV, Nc = choice_classes(mdl);
    endowCl_kjM, Nendow = endowment_classes(mdl, endow_kjM);

    frac_cekM = zeros(Nc, Nendow, mdl.K);
    for i_k = 1 : mdl.K
        # Count by [choice class, endowment class]
        cnt_ceM = zeros(Int64,  Nc, Nendow);
        for j = 1 : mdl.N
            # Endowment class for this agent
            iEndow = endowCl_kjM[i_k, j];
            # Choice class by simulated instance of hh j
            choiceClV = choiceCl_mV[choice_sjM[:, j]];
            for ic = 1 : Nc
                # Count how many simulated agents choose this choice class
                cnt_ceM[ic, iEndow] += round(Int64, sum(choiceClV .== ic));
            end
        end
        frac_cekM[:,:,i_k] = cnt_ceM ./ sum(cnt_ceM);
    end
    mStats = ModelStats(frac_cekM);
    return mStats
end


function choice_classes(mdl :: Model)
    Nendow = min(6, mdl.M);
    pctUbV = collect(range(1.0 ./ Nendow, 1.0, length = Nendow));
    choiceCl_kV = discretize_given_percentiles(collect(1.0 : mdl.M), pctUbV, false);
    return choiceCl_kV, Nendow
end


function endowment_classes(mdl :: Model, endow_kjM :: Matrix{Float64});
    Nendow = min(5, mdl.N);
    pctUbV = collect(range(1.0 ./ Nendow, 1.0, length = Nendow));
    endowCl_kjM = zeros(Int, mdl.K, mdl.N);
    for i_k = 1 : mdl.K
        endowCl_kjM[i_k, :] = discretize_given_percentiles(endow_kjM[i_k,:], pctUbV, false);
    end

    return endowCl_kjM, Nendow
end


## ------------  Deviation

function deviation(mdl :: Model,  tgStats :: ModelStats,  mStats :: ModelStats)
    dev_cekM = mStats.frac_cekM .- tgStats.frac_cekM;
    dev = sum(abs.(dev_cekM));
    mdl.currentIter += 1;

    if mdl.currentIter < 100
        showFreq = 10;
    else
        showFreq = mdl.showFreq;
    end
    if rem(mdl.currentIter, showFreq) == 0
        println("Iter $(mdl.currentIter).   Dev: $dev");
    end
    return dev :: Float64
end



# Test the objective function with random guesses
# Because NLopt returns FORCED_STOP if an error occurs
function calc_many_deviations()
    mdl = Model(showFreq = 20)
    nRuns = 400;
    # showFreq = 20;

    mp = make_random_parameters(mdl, 343);
    tgStats = solve(mdl, mp);
    dev0 = objective(mdl, tgStats, mp);

    Random.seed!(449);
    seedV = 123 .+ round.(Int,  1000 .* rand(nRuns));

    for i1 = 1 : nRuns
        mpGuess = make_random_parameters(mdl, seedV[i1]);
        devGuess = objective(mdl, tgStats, mpGuess);
        @assert abs(devGuess - dev0) > 0.001
        # guessV = mp_to_guess(mdl, mpGuess);
        # dev0 = dev_fct(guessV, []);    
        # @assert abs(dev0 - dev00) < 0.001
        # if rem(i1, showFreq) == 0
        #     println("Iteration $i1.  Deviation $dev0")
        #     # println(guessV)
        # end
    end

    return true
end


## Check that changing guesses changes solution
function check_changing_guesses(mdl)
    mp = make_random_parameters(mdl, 932);
    tgStats = solve(mdl, mp);
    guessV = mp_to_guess(mdl, mp);
    dev0 = objective(mdl, tgStats, mp);

    equalV = fill(false, n_params(mdl));
    for i1 = 1 : n_params(mdl)
        guess2V = deepcopy(guessV);
        guess2V[i1] += 0.1;
        mp2 = OptimizationLH.guess_to_mp(mdl, guess2V);
        dev2 = OptimizationLH.objective(mdl, tgStats, mp2);
        equalV[i1] = abs(dev2 - dev0) < 0.001;
    end

    success = all(equalV .== false);
    if success
        println("All guesses affect objective");
    else
        @warn "Not all guesses affect objective"
        println("Guesses that do not:");
        println(findall(equalV));
    end

    return success
end


# --------------