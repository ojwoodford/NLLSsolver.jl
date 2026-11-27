using Test, NLLSsolver, Aqua

# NLLSsolver owns optimize!/optimize and re-exports the other packages' names as const bindings (no methods
# added to functions it does not own), so no type-piracy exclusions should be needed (D2). If Aqua reports
# piracy, that is a design problem to stop and report, not to exclude away.
@testset "aqua.jl" begin
    Aqua.test_all(NLLSsolver)
end
