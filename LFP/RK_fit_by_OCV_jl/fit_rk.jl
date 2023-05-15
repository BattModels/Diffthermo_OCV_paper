module OCV
using JuMP
import Ipopt

#
# Type to Represent an RK Polynomial
#

abstract type OpenCircuitVoltage end

#Evaluate OpenCircuitVoltage types
(c::OpenCircuitVoltage)(x,T) = calcocv(c,x,T)
(c::OpenCircuitVoltage)(x) = calcocv(c,x,298.)

struct RKPolynomial{T,S} <: OpenCircuitVoltage
    Aterms::T
    U_0::S
    c_s_max::S
    c_s_min::S
    nA::Int
    # UpperFillingFraction
    # LowerFillingFraction
end

#
# Constructor for RK Polynomial
#
function RKPolynomial(Aterms,U_0,c_s_max,c_s_min)
    return RKPolynomial(Aterms,U_0,c_s_max,c_s_min,length(Aterms))
end

#
# Function to Calculate RK Polynomials
#
function calcocv(RK::RKPolynomial,x::W,T) where {W}
    F=96485.33212
    n=1.
    R=8.314
    a=1
    @inbounds rk::W = RK.Aterms[1]*((2 *x-1)^((a-1) +1))
    # Activity Correction Summation
    @simd for a in 2:RK.nA
       @inbounds @fastmath rk += RK.Aterms[a]*((2 *x-1)^((a-1) +1) - (2 *x*(a-1) *(1 -x)/ (2 *x-1)^(1 -(a-1))))
    end
    """
    For LFP, filling fraction x = 1 - SOC
    OCV = U0 + RT/nF log((1-x)/x) + RK_terms
    """
    voltage::W = @fastmath rk/(n*F) + RK.U_0 + ((R*T)/(n*F) * log((1 -x)/x)) # here, x is the filling fraction. For LFP, x = 1-SOC
    return voltage
end


# 
# Functions to force monotonicity in OpenCircuitVoltage Fits
# 
function MonotonicIncreaseLeastSquaresFit(A,y)
    model = Model(Ipopt.Optimizer)
    num_As = size(A)[2]
    dim = length(y)
    @variable(model,x[1:num_As])
    @variable(model,ŷ[1:dim])
    for i in 2:dim
        @constraint(model,ŷ[i]>=ŷ[i-1])
    end
    @constraint(model,ŷ.==A*x)
    @objective(model,Min,(ŷ.-y)'*((ŷ.-y)))
    optimize!(model)
    return model,x,ŷ
end
# 
# Functions to force monotonicity in OpenCircuitVoltage Fits
# 
function MonotonicDecreaseLeastSquaresFit(A,y)
    model = Model(Ipopt.Optimizer)
    num_As = size(A)[2]
    dim = length(y)
    @variable(model,x[1:num_As])
    @variable(model,ŷ[1:dim])
    for i in 2:dim
        @constraint(model,ŷ[i]<=ŷ[i-1])
    end
    @constraint(model,ŷ.==A*x)
    @objective(model,Min,(ŷ.-y)'*((ŷ.-y)))
    optimize!(model)
    return model,x,ŷ
end

export RKPolynomial,calcocv,MonotonicIncreaseLeastSquaresFit,MonotonicDecreaseLeastSquaresFit,get_x_from_voltage

end # module



# using OCV

using CSV, DataFrames, Tables
using JuMP
using Printf
using DelimitedFiles

function RK_Matrix(xs,nc)
    F=96485.33212        #Faraday's Constant [Coulombs/mol]
    n=1 
    R=8.314
    nr = length(xs)
    RK_mat = zeros(nr,nc+1)
    a=1
    # First Col is different
    @. RK_mat[:,1] = ((2 *xs-1)^((a-1) +1))
    # Activity Correction Summation
    for a in 2:nc
       @. RK_mat[:,a] = ((2 *xs-1)^((a-1) +1) - (2 *xs*(a-1) *(1 -xs)/ (2 *xs-1)^(1 -(a-1))))
    end
    RK_mat = RK_mat./(n.*F)
    RK_mat = hcat(RK_mat,ones(nr))
    return RK_mat
end


F=96485.33212
n=1.
R=8.314
T=298.0

# data preprocessing
filename = string("Discharge_NMat_Fig2a_even_distribution.csv")
df = DataFrame(CSV.File(filename, header=false))
# xs = 1.0 .- (df[:,1]/169.97) # filling fraction
SOC = df[:,1]/169.97 # SOC, filling fraction is 1-SOC
xs = 1.0 .- SOC
OCV_true = df[:,2] 
# # For LFP, filling fraction x = 1 - SOC
# # OCV = U0 + RT/nF log((1-x)/x) + RK_terms
ys = OCV_true - (R*T)/(n*F) * log.((1.0 .- xs)./xs) # OCV, you have to subtract the RT log term out (According to Alec)
# let xs raise
xs = reverse(xs)
ys = reverse(ys) # now ys should be monotonically decreasing
# fit
rk_order = 51
A = RK_Matrix(xs,rk_order) 
a = OCV.MonotonicDecreaseLeastSquaresFit(A,ys)
xs = reverse(xs) # after fitting, reverse back xs and ys
ys = reverse(ys) # now ys should be monotonically increasing
rk_params = value.(a[1][:x]) # value function comes from JuMP pkg
U_0 = rk_params[length(rk_params)]   # the last fitted param is U0
@printf("U0 fitted is  %.4f V\n", (U_0))

# calculate fitted OCV
c_s_max = 100000.0
c_s_min = 0.0
cathodeocv = OCV.RKPolynomial(rk_params[1:length(rk_params)-1],U_0,c_s_max,c_s_min)
atol = 1e-10
OCV_predicted = zeros(length(xs))
RMSE = 0.0
for i in range(length(xs), step=-1, stop=1)
    x = xs[i]
    V = OCV.calcocv(cathodeocv,x,298)
    global OCV_predicted[i] = V
    # print(V, "   ", OCV_true[i], "\n")
    global RMSE = RMSE + (V-OCV_true[i])^2
end
# print("\n\n\n")
# print(OCV_predicted)
# print("\n\n\n")
# RMSE = sqrt(RMSE/length(xs))
# @printf("RMSE = %.4f V", (RMSE))
# print("\n")

CSV.write("PredictedOCV.csv",  Tables.table(hcat(SOC, OCV_predicted)), writeheader=false)

