###########################################################
#### To calculate Lyapunov exponents.
#### The data is galaxyHalo simulated by Gadget.
#### For actions and chaos.
###########################################################

#### preparing
function read_txt_float(filename; line_by_each=1)
    arr = Any[]
    # arr = Array{Float64, 2} #[learn code] this can not push!()
    line = 0
    open(filename, "r") do f
        while (!eof(f) & true)
            # read a new / next line for every iteration          
            s = readline(f)
            b = split(s)
            arr1 = Any[]
            for b1 in b
                push!(arr1, parse(Float64, b1))
            end
            # println(arr1)
            push!(arr, arr1)
            line += line_by_each #??
            # println("$line . $s")
        end
    end
    eachline = length(arr[1])
    arr_rt = Array{Float64, 2}(undef, line, eachline)
    for i in 1:line
        arr_rt[i,:] = Float64.(arr[i])
    end
    return arr_rt
end



#### main
#
## section
filename = "particle_1.txt"
filename = "particle_2.txt"
data = read_txt_float(filename)
Dim = 3
data = data[:,1:Dim]

using DynamicalSystems, CairoMakie
ds = Systems.henon()
# data = DynamicalSystems.ChaosTools.trajectory(ds, 10000)
ksend = 20
lendata = Int32(size(data)[1])
# lendata = length(data)
println("lendata: ", lendata)

ks = 1:ksend
R_hw_end = Int32((lendata-ksend)*0.9)
R_hw = 1:R_hw_end
println("R_hw: ", R_hw)
# exit(1)

fig = Figure(figsize=(500,500))
# for (i, di) in enumerate([Euclidean(), Cityblock()])
#     ax = Axis(fig[1, i]; title = "Distance: $(di)")#, fontsize = 18)
#     x = data[:, 3] # fake measurements for the win!

for j in 1:Dim
    di = Euclidean()#, Cityblock()])
    ax = Axis(fig[j,1]; title = "dimension: $(j)")#, fontsize = 18)
    x = data[:, j] # fake measurements for the win!

    ntype = NeighborNumber(2)
    for D in [2, 4, 7]
        R = embed(x, D, 1)
        # println("R: ", R)
        E = lyapunov_from_data(R, ks;
            refstates = R_hw, distance = di, ntype = ntype)
        Δt = 1
        λ = linear_region(ks.*Δt, E)[2]
        # gives the linear slope, i.e. the Lyapunov exponent
        lines!(ax, ks .- 1, E .- E[1], label = "D=$D, λ=$(round(λ, digits = 3))")
    end
    axislegend(ax)
end
fig
#

#=
## section
using DynamicalSystems, CairoMakie
henon = Systems.henon()
N_pp = 100
tr1 = trajectory(henon, N_pp)
smr = summary(tr1)
println("tr1: ", smr)

u2 = get_state(henon) + (1e-9 * ones(dimension(henon)))
tr2 = trajectory(henon, N_pp, u2)
smr = summary(tr2)
println("tr1: ", smr)

using LinearAlgebra: norm
fig = Figure()

# Plot the x-coordinate of the two trajectories:
ax1 = Axis(fig[1,1]; ylabel = "x")
lines!(ax1, tr1[:, 1])#; color = Cycled(1))
lines!(ax1, tr2[:, 1])#; color = (Main.COLORS[2], 0.5))
hidexdecorations!(ax1; grid = false)

# Plot their distance in a semilog plot:
ax2 = Axis(fig[2,1]; ylabel = "d", xlabel = "n", yscale = log)
d = [norm(tr1[i] - tr2[i]) for i in 1:length(tr2)]
lines!(ax2, d; color = Cycled(3))
println(fig)
fig #[learn code] it might does not need PyPlot.show()
=#

#=
## section
using DynamicalSystems
using LinearAlgebra, PyPlot
lo = Systems.lorenz([20,20,20.0])
λ = DynamicalSystems.ChaosTools.lyapunov(lo, 100000) #lyap
plot([4, 19], λ.*[0, 15] .- 13)
X₁ = trajectory(lo, 45)
u₂ = get_state(lo) .+ 1e-6
X₂ = trajectory(lo, 45, u₂)
δ = norm.(X₂.data .- X₁.data) #d_orb
plot(0:0.01:45, log.(δ))
PyPlot.show()
=#

#=
## section
using DynamicalSystems
using LatexPrint

function lorenz_rule(u, p, t) # the dynamics as a function
    sigma, rho, beta = p
    x, y, z = u
    dx = sigma*(y - x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
    return SVector(dx, dy, dz) # Static Vector
end

p = [10.0, 28.0, 8/3] # parameters: σ, ρ, β
u_0 = [0.0, 10.0, 0.0] # initial condition
# create an instance of a `DynamicalSystem`
lorenz = ContinuousDynamicalSystem(lorenz_rule, u_0, p)
println(lorenz)

T = 100.0 # total time
delta_t = 0.01 # sampling time
A = trajectory(lorenz, T)
# A = trajectory(lorenz, T; delta_t) #??
println(A)
=#
