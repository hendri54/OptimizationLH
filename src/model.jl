## -------------  Model

# Make random model parameters for testing and intialization
# All in [1, 2]
# Changes GLOBAL_RNG
function make_random_parameters(mdl :: Model, randSeed :: Integer)
    Random.seed!(randSeed);
    return ModelParams(rand(n_params(mdl)) .+ 1.0)
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
    @assert length(guessV) == n_params(mdl)
    return ModelParams(guessV)
end


function mp_to_guess(mdl :: Model, mp :: ModelParams)
    return mp.alphaV
end


## -----------  Solve

function solve(mdl :: Model, mp :: ModelParams)
    # endow_jV = make_endowments(mdl,  mp);
    # optValue_mV = make_option_values(mdl, mp);

    prob_mjM = decision_probabilities(mdl, mp);
    choice_sjM = simulate_choices(mdl, prob_mjM, 320);
    mStats = compute_stats(mdl, choice_sjM);

    return mStats
end


## Make endowments of households
function make_endowments(mdl :: Model)

    # Random.seed!(583);
    # outV = rand(mdl.N, mdl.K + 1) * vcat(1.0, mp.alphaV);
    outV = collect(1 : mdl.N) ./ mdl.N;
    return outV :: Vector{Float64}
    # endow_kjM = zeros(mdl.K, nInd);
    # endowV = collect(1 : nInd) ./ nInd .* 2;
    # for k = 1 : mdl.K
    #     endow_kjM[k,:] = (endowV .^ 0.6) .* (alphaV[k] .+ betaV[k] .* endowV);
    # end
    # return endow_kjM
end


## Option value = fixed matrix * alphaV
function make_option_values(mdl :: Model, mp :: ModelParams)
    Random.seed!(439);
    outV = rand(mdl.M, mdl.K + 1) * vcat(1.0, mp.alphaV);
    return outV :: Vector{Float64}
end


## Decision probabilities
function decision_probabilities(mdl :: Model, mp :: ModelParams)
    v_mV = make_option_values(mdl, mp);
    value_jV = make_endowments(mdl);
    # The trick is to find a bounded function with many parameters
    util_mjM = exp.(sin.(5.0 .* v_mV * transpose(value_jV)));
    prob_mjM = util_mjM ./ sum(util_mjM, dims = 1);
    # prob_mjM = zeros(mdl.M, mdl.N);
    # for j = 1 : mdl.N
    #     prob_mjM[:, j] = decision_prob(mdl, endow_jV[j],  optValue_mV);
    # end

    @assert all(prob_mjM .> 0.0001)
    return prob_mjM
end


## Decision probabilties for one household
# Inputs are endowments and option characteristics
# Must be nonlinear. Otherwise some parameters do not affect decision probs (intercepts)
# function decision_prob(mdl :: Model, endow :: Float64, optValue_mV :: Vector{Float64})
#     # Utility of alternatives
#     utilV = zeros(mdl.M);
#     for m = 1 : mdl.M
#         # Ensures positive utility
#         utilV[m] = sum((endow_kV .* optValue_kmM[:, m]) .^ 0.5);
#     end
#     @assert all(utilV .> 0.0)
#     @assert all(utilV .< 1e5)
#     probV = utilV ./ sum(utilV);
#     @assert all(probV .> 0.0)
#     @assert all(probV .< 1.0)
#     return probV
# end


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
function compute_stats(mdl :: Model,  choice_sjM :: Matrix{T1})  where T1 <: Integer

    choiceCl_mV, Nc = choice_classes(mdl);
    endowCl_jV, Nendow = endowment_classes(mdl);

    # Count by [choice class, endowment class]
    cnt_ceM = zeros(Int64,  Nc, Nendow);
    for j = 1 : mdl.N
        # Endowment class for this agent
        iEndow = endowCl_jV[j];
        # Choice class by simulated instance of hh j
        choiceClV = choiceCl_mV[choice_sjM[:, j]];
        for ic = 1 : Nc
            # Count how many simulated agents choose this choice class
            cnt_ceM[ic, iEndow] += round(Int64, sum(choiceClV .== ic));
        end
    end
    frac_ceM = cnt_ceM ./ sum(cnt_ceM);
    mStats = ModelStats(frac_ceM);
    return mStats
end


function choice_classes(mdl :: Model)
    Nendow = min(mdl.nChoiceCl, mdl.M - 2);
    pctUbV = collect(range(1.0 ./ Nendow, 1.0, length = Nendow));
    choiceCl_kV = discretize_given_percentiles(collect(1.0 : mdl.M), pctUbV, false);
    return choiceCl_kV, Nendow
end


function endowment_classes(mdl :: Model);
    Nendow = min(mdl.nEndowCl, mdl.N - 2);
    endow_jV = make_endowments(mdl);
    pctUbV = collect(range(1.0 ./ Nendow, 1.0, length = Nendow));
    endowCl_jV = discretize_given_percentiles(endow_jV, pctUbV, false);

    return endowCl_jV, Nendow
end


## ------------  Deviation

function deviation(mdl :: Model,  tgStats :: ModelStats,  mStats :: ModelStats)
    dev_ceM = mStats.frac_ceM .- tgStats.frac_ceM;
    dev = sum(abs.(dev_ceM));
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