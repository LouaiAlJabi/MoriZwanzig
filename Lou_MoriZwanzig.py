import mori_zwanzig as mz
import numpy as np
import pickle
import time as tm
from sklearn.linear_model import LinearRegression
import concurrent.futures as cf

def LoadDataFiles(*args):
    return [np.load(i) for i in args]

def VarSetter(*args):
    return [i for i in args]

def PickleResults(name,result):
    with open(name,'wb') as file:
        pickle.dump(result,file,protocol=pickle.HIGHEST_PROTOCOL)

def PackData(*args):
    dataTuple = np.concatenate([i for i in args],axis=2)
    return dataTuple

def AutoCorrelationFunctions(velocity,Force):
    """
    This function calculates correlation functions using the provided velocities and forces

    args:
    velocity = the file of velocities
    Froce = the file of forces
    """
    start = tm.time()
    with cf.ProcessPoolExecutor() as exe:
        future_vacf = exe.submit(mz.compute_correlation_functions, velocity,None,True,False,False)
        future_Facf = exe.submit(mz.compute_correlation_functions, Force,None,True,False,False)
        future_vFcross = exe.submit(mz.compute_correlation_functions, velocity,Force,True,False,False)
    
        v_acf = future_vacf.result()
        F_acf = future_Facf.result()
        vF_cross = future_vFcross.result()
    
    
    print(f"multiprocessing took {(round((tm.time()-start)/60,2))} mins to finish correlation functions")    
    return [v_acf, F_acf, vF_cross]

def diffusion_constant_from_MSD(start_fit_time, end_fit_time,time,msd): #24.,28.
    start_fit_index = np.argmin(np.abs(time - start_fit_time))
    end_fit_index = np.argmin(np.abs(time - end_fit_time))
    model = LinearRegression()
    model.fit(time[start_fit_index:end_fit_index].reshape((-1,1)), 
                msd[start_fit_index:end_fit_index])
    m = model.coef_[0]
    D = m/2
    return D

def BrownianParticleDict(dataStructure, part_mass,dt,KB,T,Omega,start_fit_time,end_fit_time,time,cutoffs,num_steps):
    v_acf = dataStructure[:,:,:3]
    F_acf = dataStructure[:,:,3:6]
    vF_cross = dataStructure[:,:,6:9]
    sq_displacement = dataStructure[:,:,9:]

    finalResults = []
    
    results = dict()
    results['Einstein'] = 0
    results['Direct'] = 0
    results["v_acf_boot"] = np.array([])
    results["F_acf_boot"] = np.array([])
    results["vF_cross_boot"] = np.array([])
    results['K_Values'] = np.array([])
    results['K1_Values'] = np.array([])
    results['K3_Values'] = np.array([])
    results['MZ'] = dict()
    results['v_acf_mz'] = dict()
    
    for cutoff in cutoffs:
        results["MZ"][cutoff] = 0

    for cutoff in cutoffs:
        results["v_acf_mz"][cutoff] = np.array([])

    v_acf_boot = np.mean(v_acf, axis=(1,2))
    F_acf_boot = np.mean(F_acf, axis=(1,2))
    vF_cross_boot = np.mean(vF_cross, axis=(1,2))
    results["v_acf_boot"] = v_acf_boot
    results["F_acf_boot"] = F_acf_boot
    results["vF_cross_boot"] = vF_cross_boot
    
    msd = np.mean(sq_displacement, axis=(1,2))
    print("Finished the means")
        
    K1 = -F_acf_boot / (part_mass * KB * T)
    K3 = vF_cross_boot / (KB * T)
    K = mz.get_K_from_K1_and_K3(dt, K1, K3)
    C0 = v_acf_boot[0]
    results['K_Values'] = np.array([K])
    results['K1_Values'] = K1
    results['K3_Values'] = K3
    print("Finshed K stuff. Check point before mz")
        
    for cutoff in cutoffs:
        v_acf_cutoff = mz.integrate_C(dt, num_steps, Omega, K, C0 = C0, cutoff = cutoff)
        results['MZ'][cutoff] = np.trapz(v_acf_cutoff, time)
        results['v_acf_mz'][cutoff] = v_acf_cutoff
    print("Another check point after mz")

    
    results['Einstein'] = diffusion_constant_from_MSD(start_fit_time,end_fit_time,time,msd)
    results['Direct'] = np.trapz(v_acf_boot,time)
    print("Done")
    return results

def BrownianParticleList(dataStructure, part_mass,dt,KB,T,Omega,start_fit_time,end_fit_time,time,cutoffs,num_steps):
    v_acf = dataStructure[:,:,:3]
    F_acf = dataStructure[:,:,3:6]
    vF_cross = dataStructure[:,:,6:9]
    sq_displacement = dataStructure[:,:,9:]

    finalResults = []
    results = dict()
    results['Einstein'] = 0
    results['Direct'] = 0
    results["v_acf_boot"] = np.array([])
    results["F_acf_boot"] = np.array([])
    results["vF_cross_boot"] = np.array([])
    results['K_Values'] = np.array([])
    results['K1_Values'] = np.array([])
    results['K3_Values'] = np.array([])
    results['MZ'] = dict()
    results['v_acf_mz'] = dict()
    
    for cutoff in cutoffs:
        results["MZ"][cutoff] = 0

    for cutoff in cutoffs:
        results["v_acf_mz"][cutoff] = np.array([])

    v_acf_boot = np.mean(v_acf, axis=(1,2))
    F_acf_boot = np.mean(F_acf, axis=(1,2))
    vF_cross_boot = np.mean(vF_cross, axis=(1,2))
    results["v_acf_boot"] = v_acf_boot
    results["F_acf_boot"] = F_acf_boot
    results["vF_cross_boot"] = vF_cross_boot
    
    msd = np.mean(sq_displacement, axis=(1,2))
    print("Finished the means")
        
    K1 = -F_acf_boot / (part_mass * KB * T)
    K3 = vF_cross_boot / (KB * T)
    K = mz.get_K_from_K1_and_K3(dt, K1, K3)
    C0 = v_acf_boot[0]
    results['K_Values'] = K
    results['K1_Values'] = K1
    results['K3_Values'] = K3
    print("Finshed K stuff. Check point before mz")
        
    for cutoff in cutoffs:
        v_acf_cutoff = mz.integrate_C(dt, num_steps, Omega, K, C0 = C0, cutoff = cutoff)
        results['MZ'][cutoff] = np.trapz(v_acf_cutoff, time)
        results['v_acf_mz'][cutoff] = v_acf_cutoff
    print("Another check point after mz")

    
    results['Einstein'] = diffusion_constant_from_MSD(start_fit_time,end_fit_time,time,msd)
    results['Direct'] = np.trapz(v_acf_boot,time)
    print("Done")
    
    for key, value in results.items():
        finalResults.append(value)
    
    finalResults = tuple(finalResults)    
    return finalResults