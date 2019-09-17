module OptimizationLH

using Distributions, Formatting, Parameters, Random
using NLopt

export Model, calibrate, make_endowments, decision_probabilities
export make_random_parameters

include("discretize.jl")
include("types.jl")
include("model.jl")



"""
    calibrate()

Calibrate the model
"""
function calibrate(mdl :: Model, algo :: Symbol)
    # algo = :LN_COBYLA;
    # algo = :GN_CRS2_LM;
    # algo = :LN_SBPLX;

    # Solve with random parameters to make attainable calibration targets
    mpTrue = make_random_parameters(mdl, 392);
    solnV = mp_to_guess(mdl, mpTrue);

    # Calibration targets = model solution
    tgStats = solve(mdl, mpTrue);

    # Objective should recover tgStats
    dev = objective(mdl, tgStats, mpTrue);
    @assert isa(dev, Float64);
    @assert dev ≈ 0.0

    # Objective wrapper to be called by solver
    function dev_fct(guessInV :: Vector{Float64}, gradV)
        @assert length(gradV) == 0
        mpDev = guess_to_mp(mdl, guessInV);
        return objective(mdl, tgStats, mpDev);
    end

    # Check that wrapper returns the same as calling objective directly
    dev2 = dev_fct(solnV, []);
    @assert dev2 ≈ dev

    ## ------  Optimization

    nParams = n_params(mdl);
    println("\n--------------------")
    println("Starting $algo with $nParams parameters");
    nTgMoments = n_tg_moments(tgStats);
    println("  No of target moments: $nTgMoments")

    mpGuess = make_random_parameters(mdl, 493);
    guessV = mp_to_guess(mdl, mpGuess);
    dev0 = dev_fct(guessV, []);
    println("Initial deviation: $dev0")

    
    optS = Opt(algo,  nParams);
    optS.min_objective = dev_fct;
    optS.lower_bounds = 1.0;
    optS.upper_bounds = 2.0;
    optS.stopval = 0.001;
    optS.stopval = 0.001;
    optS.ftol_rel = 0.001;
    optS.maxeval = 4_000;
    mdl.currentIter = 0;
    optf, optx, ret = optimize(optS, guessV);

    optfr = round(optf, digits = 3);
    nIter = mdl.currentIter;
    println("Optimization complete.   $nIter iterations.   f = $optfr");
    println("  Return value: $ret");
    mp = guess_to_mp(mdl, optx);
    devSol = objective(mdl, tgStats, mp);
    @assert abs(devSol - optf) < 0.001

    meanAbsDev = round(mean(abs.(optx .- solnV)), digits = 3);
    println("  Mean abs deviation vs true solution x: $meanAbsDev");
    println("----- Initial parameters:")
    show_params(mpTrue);
    println("----- Solution parameters:")
    show_params(mp);

    return nothing
end



end # module
