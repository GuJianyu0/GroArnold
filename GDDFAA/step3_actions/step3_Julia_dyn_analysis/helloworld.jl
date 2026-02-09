###########################################################
## The packages and files to be included.

## something
# julia documentation
# https://docs.julialang.org/en/v1/

# julia DynamicalSystems
# https://juliadynamics.github.io/DynamicalSystems.jl/stable/

# lyapunovs
# https://juliadynamics.github.io/DynamicalSystems.jl/stable/chaos/lyapunovs/#ChaosTools.lyapunov_from_data

# About the network, add one line in /etc/julia/startup.jl or julia/etc/... file.
# ENV["JULIA_PKG_SERVER"] = "https://mirrors.tuna.tsinghua.edu.cn/julia"

# change versions
# https://blog.csdn.net/m0_37952030/article/details/108966452?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-108966452-blog-122006098.pc_relevant_aa&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-108966452-blog-122006098.pc_relevant_aa&utm_relevant_index=2
###########################################################



## Hello world
# a = 1
# function func1(a)
#     return a
# end

# print("Hello world, ", a,"\n")
# print(func1(3.9),"\n")



# packages
using Pkg
Pkg.status()

Pkg.add("LinearAlgebra")
using LinearAlgebra
# Pkg.add("StochasticDiffEq")
# using StochasticDiffEq
Pkg.add("DifferentialEquations")
using DifferentialEquations
Pkg.add("CairoMakie")
using CairoMakie

Pkg.add("DynamicalSystems")
Pkg.build("FFTW")
using DynamicalSystems
# DynamicalSystems
# #\[learn code] solve error one by one when precompile
# #\[learn code] it will cost much time when using this pkg

Pkg.add("LatexPrint")
using LatexPrint

# Pkg.add("Plots")
# using Plots
Pkg.add("PyPlot")
using PyPlot

println("Done.")



## bugs
# signal (11): Segmentation fault
# in expression starting at none:0
# jl_subtype_env at /usr/bin/../lib/x86_64-linux-gnu/libjulia.so.1 (unknown line)
# jl_isa at /usr/bin/../lib/x86_64-linux-gnu/libjulia.so.1 (unknown line)
# rewrap at ./compiler/typeutils.jl:8 [inlined]
# matching_cache_argtypes at ./compiler/inferenceresult.jl:132
# InferenceResult at ./compiler/inferenceresult.jl:12 [inlined]
# InferenceResult at ./compiler/inferenceresult.jl:12 [inlined]
# typeinf_ext at ./compiler/typeinfer.jl:572
# typeinf_ext at ./compiler/typeinfer.jl:605
# jfptr_typeinf_ext_1.clone_1 at /usr/lib/x86_64-linux-gnu/julia/sys.so (unknown line)
# unknown function (ip: 0x7f433c3e1410)
