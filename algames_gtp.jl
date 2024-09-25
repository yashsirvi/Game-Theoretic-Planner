using Algames
using AlgamesDriving
using StaticArrays
using LinearAlgebra

p = 3
dt = 0.1
solver_opts = Options()
        

model = BicycleGame(p=p)    