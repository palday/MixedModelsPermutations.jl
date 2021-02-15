
#using DataFrames, CSV, Pipe, QuantEcon, StatsModels, LinearAlgebra, MixedModels, Random, Plots
# removed QuantEcon from requirements
using DataFrames,CSV,StatsModels,LinearAlgebra,MixedModels,Random

#include("test/expandgrid.jl")

#include("test/exponentialCorrelation.jl")
#include("test/Circulant.jl")
#include("test/fitsignal.jl")
include("test/sim_utilities.jl")

n_part = 20 # participants
n_item = 30 # items
n_t = 400

Σ = circulant(exponentialCorrelation([0:1:(n_t-1);],nu=1))
Ut =  LinearAlgebra.cholesky(Σ).U'


df_item = DataFrame(id_item = 1:n_item)
df_part = DataFrame(id_part = 1:n_part)
#df1 = df_part
#df2 = df_item

transform!(df_part, :id_part=> (x->ifelse.(x .< (n_part/2),"Fp1","Fp2")) => :f_part)
transform!(df_item, :id_item=> (x->ifelse.(x .< (n_item/2),"Fi1","Fi2")) => :f_item)


df = expandgrid(df_part,df_item)





### add random intercept participant
df.rep = map(x -> randn(Random.MersenneTwister(x + 1000),Float64,(1,n_t))*Ut',df.id_part)


### add random slope participant / different per item-category
df.repf = map((x,y) -> randn(Random.MersenneTwister(x + 100*y+ 10000),Float64,(1,n_t))*Ut',df.id_part,df.f_item.=="Fi1")

### add random intercept item
df.rei = map(x -> randn(Random.MersenneTwister(x + 20000),Float64,(1,n_t))*Ut',df.id_item)

### add random slope item ( different per subject-category)
df.reif = map((x,y) -> randn(Random.MersenneTwister(x + 100*y+ 30000),Float64,(1,n_t))*Ut',df.id_item,df.f_part.=="Fp1")



### add error
df.err = map(x -> randn(Random.MersenneTwister(x +40000),Float64,(1,n_t))*Ut',[1:1:nrow(df);])


df.y = df.rep + df.repf + df.rei + df.reif + df.err
df.part = categorical(df.id_part)
df.item = categorical(df.id_item)
df = select(df,[:part, :f_part, :item, :f_item, :y])

#sign = reduce(vcat,df.y)
# more juliaesk:
sign = vcat(df.y...)



### models
co = Dict(:f_part => EffectsCoding(), :f_item => EffectsCoding())
form = @formula(yi ~ f_part*f_item + (1|part)+ (1|part&f_item)+ (1|item) + (1|item&f_part))

lmods = fitsignal(form,df,sign,co)

ff = @formula(0~f_part*f_item)
mm = ModelMatrix(ModelFrame(ff, df, contrasts = co))


