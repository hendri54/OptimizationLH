using OptimizationLH
using Test

@testset "Model" begin
    mdl = Model();

    # Helpers for computing stats
    choiceCl_kV, Nc = OptimizationLH.choice_classes(mdl);
    @test all(choiceCl_kV .>= 1)
    @test all(choiceCl_kV .<= Nc)
    @test isa(choiceCl_kV, Vector{Int})

    mp = OptimizationLH.make_random_parameters(mdl, 215);
    @test length(mp.alphaV) == mdl.K

    # Handling guesses
    guessV = OptimizationLH.mp_to_guess(mdl, mp);
    @test length(guessV) == 4 * mdl.K
    @test isa(guessV, Vector{Float64})
    mp2 = OptimizationLH.guess_to_mp(mdl, guessV);
    @test mp2.alphaV == mp.alphaV
    guess2V = OptimizationLH.mp_to_guess(mdl, mp2);
    @test guessV == guess2V

    # Simulation
    prob_mjM = rand(mdl.M, mdl.N);
    choice_sjM = OptimizationLH.simulate_choices(mdl, prob_mjM ./ sum(prob_mjM, dims = 1), 242);
    @test isa(choice_sjM, Matrix{Int})
    @test size(choice_sjM) == (mdl.nSim, mdl.N)
    @test all(choice_sjM .>= 1)
    @test all(choice_sjM .<= mdl.M)

    # Solving model
    mStats = OptimizationLH.solve(mdl, mp);

    # Check that perturbing each guess changes objective
    @test OptimizationLH.check_changing_guesses(mdl);

    @test OptimizationLH.calc_many_deviations();

    # Calibration
    calibrate(mdl, :LN_SBPLX)
end


# --------------