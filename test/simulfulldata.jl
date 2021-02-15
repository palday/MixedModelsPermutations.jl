

# Pkg.add("Pipe")
# Pkg.add("DataFrames")
# Pkg.add("QuantEcon")
# Pkg.add("StatsModels")
# Pkg.add("CSV")
# Pkg.add("MixedModels")

using DataFrames, CSV, Pipe, QuantEcon, StatsModels, LinearAlgebra, MixedModels, Random, Plots

# include("download_titantic.jl")
#titan = download_titantic()
#@pipe titan|>
#    filter(:sex => ==("male"),_)

npart = 20
nitem = 30
nt = 400
 include("test/expandgrid.jl")
 include("test/exponentialCorrelation.jl")
 include("test/Circulant.jl")
 include("test/fitsignal.jl")

 Sigma = Circulant(exponentialCorrelation([0:1:(nt-1);],nu=1))
Ut =  LinearAlgebra.cholesky(Sigma).U'


df_item = DataFrame(iditem = 1:nitem)
df_part = DataFrame(idpart = 1:npart)
df1 = df_part
df2 = df_item

transform!(df_part, :idpart=> (x->ifelse.(x .< (npart/2),"Fp1","Fp2")) => :fpart)
transform!(df_item, :iditem=> (x->ifelse.(x .< (nitem/2),"Fi1","Fi2")) => :fitem)

df = expandgrid(df_part,df_item)





### add random intercept particiapnt
df.rep = map(x -> randn(Random.MersenneTwister(x + 1000),Float64,(1,nt))*Ut',df.idpart)


### add random slope particiapnt fitem
df.repf = map((x,y) -> randn(Random.MersenneTwister(x + 100*y+ 10000),Float64,(1,nt))*Ut',df.idpart,df.fitem.=="Fi1")

### add random intercept item
df.rei = map(x -> randn(Random.MersenneTwister(x + 20000),Float64,(1,nt))*Ut',df.iditem)

### add random slope item fpart
df.reif = map((x,y) -> randn(Random.MersenneTwister(x + 100*y+ 30000),Float64,(1,nt))*Ut',df.iditem,df.fpart.=="Fp1")



### add error
df.err = map(x -> randn(Random.MersenneTwister(x +40000),Float64,(1,nt))*Ut',[1:1:nrow(df);])

##### to string
df =
    @pipe  df|>
    transform(_, :idpart =>(x-> x=map(string,repeat("p",length(x)),x)) => :part)|>
    transform(_, :iditem =>(x-> x=map(string,repeat("p",length(x)),x)) => :item)



#transform!(df, [:rep,:repf,:rei,:reif,:err] => .+ => :y)

df.y = df.rep + df.repf + df.rei + df.reif + df.err

df =
    @pipe  df|>
    select(_,[:part, :fpart, :item, :fitem, :y])

sign = reduce(vcat,df.y)

### models


co = Dict(:fpart => EffectsCoding(), :fitem => EffectsCoding())
form = @formula(yi ~ fpart*fitem + (1|part)+ (1|part&fitem)+ (1|item) + (1|item&fpart))

#df[:,formula.lhs.sym] = (signal[:,1])
#mod1 = MixedModels.fit(MixedModel, formula, df, contrasts =contrasts)


lmods = fitsignal(form,df,sign,co)

ff = @formula(0~fpart*fitem)
mm = ModelMatrix(ModelFrame(ff, df, contrasts = co))
fm = @formula(yield ~ + f+ (1|batch))

