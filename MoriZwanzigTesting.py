import Lou_MoriZwanzig as lmz
import numpy as np
import Lou_GeneralBootstrap as lbs

velocity, force, sq_displacement = lmz.LoadDataFiles("velocities.npy", "forces.npy", "sq_displacement_stat.npy")
newVel = np.array_split(velocity,20,1)[0]
newforce = np.array_split(force,20,1)[0]
newSq = np.array_split(sq_displacement,20,1)[0]
vel_acf, Force_acf, velF_cross = lmz.AutoCorrelationFunctions(newVel,newforce) 
dataTuple = lmz.PackData(vel_acf, Force_acf, velF_cross,newSq)

num_steps, num_sims, num_dimensions = newVel.shape
part_mass, dt,KB,T,Omega,start_fit_time,end_fit_time = lmz.VarSetter(80.,0.005,1.,1.,0,24.,28.)
time = np.linspace(0, dt*num_steps, num_steps)
cutoffs = np.arange(1,29.)

singleBrownian = lmz.BrownianParticleList(dataTuple,part_mass, dt,KB,T,Omega,start_fit_time,end_fit_time,time,cutoffs,num_steps)
fullBoot = lbs.Bootstrap(lmz.BrownianParticleList,dataTuple,8,1,4,1,True,part_mass = part_mass, dt=dt,KB=KB,T=T,Omega=Omega,start_fit_time=start_fit_time,end_fit_time=end_fit_time,time=time,cutoffs=cutoffs,num_steps=num_steps)