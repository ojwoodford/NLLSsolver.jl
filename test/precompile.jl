using Test, NLLSsolver

# A package that calls optimize! in a PrecompileTools workload must precompile: evaluating into NLLSsolver (as getiterator once
# did) is an error while another package is being precompiled
@testset "Precompile workload" begin
    mktempdir() do dir
        pkgdir = joinpath(dir, "NLLSPrecompileCheck")
        mkpath(joinpath(pkgdir, "src"))
        write(joinpath(pkgdir, "Project.toml"), """
            name = "NLLSPrecompileCheck"
            uuid = "5d0a8b2e-6c1f-4a53-9f0e-2b7c3d1e4a96"
            version = "0.1.0"

            [deps]
            NLLSsolver = "50b5cf9e-bf7a-4e29-8981-4280cbba70cb"
            PrecompileTools = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
            Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
            StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
            """)
        write(joinpath(pkgdir, "src", "NLLSPrecompileCheck.jl"), """
            module NLLSPrecompileCheck
            using NLLSsolver, PrecompileTools, Static, StaticArrays

            struct Offset <: AbstractResidual end
            NLLSsolver.ndeps(::Offset) = static(2)
            NLLSsolver.nres(::Offset) = static(1)
            NLLSsolver.varindices(::Offset) = SVector(1, 2)
            NLLSsolver.getvars(::Offset, vars::Vector) = (vars[1]::Float64, vars[2]::Float64)
            NLLSsolver.computeresidual(::Offset, a, b) = SVector(b - a - 1.0)
            Base.eltype(::Offset) = Float64

            @setup_workload begin
                @compile_workload begin
                    problem = NLLSProblem(Float64, Offset)
                    addvariable!(problem, 0.0)
                    addvariable!(problem, 3.0)
                    addcost!(problem, Offset())
                    optimize!(problem)
                end
            end
            end
            """)
        # Precompile and load the package in a fresh process, with the same environment as this one
        script = "push!(LOAD_PATH, $(repr(dir))); using NLLSPrecompileCheck"
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`
        out = IOBuffer()
        @test success(pipeline(cmd; stdout=out, stderr=out))
        @test !occursin("breaks incremental compilation", String(take!(out)))
    end
end
