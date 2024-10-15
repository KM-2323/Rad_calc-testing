import numpy as np
import scf as mscf
import cisd as cisd
import util as util
from rad_settings_opt import *
import sys

def rad_calc(file,params):
    #Call HF driver--Comment out last line to stop here
    coord,atoms_array,coord_w_h,dist_array,nelec,ndocc,n_list,natoms_c,natoms_n,natoms_cl,energy0,one_body,two_body,orb_energy,hf_orbs,fock_mat=mscf.main_scf(file,params)
    com,coord = util.re_center(coord,atoms_array,coord_w_h)
    hf_orbs = util.orb_sign(hf_orbs,orb_energy,nelec,dist_array,alt)
    print("\n--------------------------")
    print("Converged ROPPP Orbitals")
    print("--------------------------\n")
    natoms=np.shape(coord)[0]
    for iorb in range(natoms):
        print('orbital number', iorb + 1, 'energy', orb_energy[iorb]-orb_energy[int((nelec-1)/2)])
        print(np.around(hf_orbs[:, iorb], decimals=2)) #print(np.around(guess_orbs[:, iorb], decimals=2))

            #########################################################
             # PRINTING OF MOLECULAR ORBITALS BASED ON GAMESS OUTPUT #
             #########################################################
    atomic_numbers=[]
    for atom in atoms_array:
        number={"C":6.0,"c":6.0,"H":1.0,"h":1.0,"N":7.0,"n":7.0,"N1":7.0,"n1":7.0,"N2":7.0,"n2":7.0,"Cl":17.0,"cl":17.0,"CL":17.0}[atom[0]]
        atomic_numbers.append([atom[0],number])
    f=open('%s.out'%file,'w')
    f.write("\n")
    f.write("\nGAMESS COORDINATES FORMAT")
    f.write("\n")
    f.write("\n ATOM      ATOMIC                      COORDINATES (BOHR)")
    f.write("\n           CHARGE         X                   Y                   Z")
    #for i,atom in enumerate(atoms_array):
    for i in range(natoms_c+natoms_n+natoms_cl):
        f.write("\n %s           %d     %f            %f            %f"%(atoms_array[i][0],atomic_numbers[i][1],coord[i,0]*tobohr,coord[i,1]*tobohr,coord[i,2]*tobohr))
    f.write("\n                      ")
    f.write("\n     ATOMIC BASIS SET")
    f.write("\n     ----------------")
    f.write("\n ")
    f.write("\n ")
    f.write("\n ")
    f.write("\n  SHELL TYPE  PRIMITIVE        EXPONENT          CONTRACTION COEFFICIENT(S)")
    f.write("\n ")
    n1=1
    n2=1
    for i,atom in enumerate(atoms_array):
        if atom[0] == 'C':
            f.write("\n C         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s            27.3850330    0.430128498301"%(str(n1+i),str(n2+i)))
            f.write("\n     %2s   S     %3s             4.8745221    0.678913530502"%(str(n1+i),str(n2+i+1)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             1.1367482    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             0.2883094    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            n1+=1
            n2+=3
        if atom[0] == 'Cl':
            f.write("\n Cl         ")
            f.write("\n ")
            f.write("\n     %2s   S     %3s           229.9441039    0.430128498301"%(n1+i,n2+i))
            f.write("\n     %2s   S     %3s            40.9299346    0.678913530502"%(n1+i,n2+i+1))
            f.write("\n ")
            f.write("\n     %2s   L     %3s            15.0576101    0.049471769201    0.511540707616"%(str(n1+i+1),str(n2+i+2)))
            f.write("\n     %2s   L     %3s             3.8190075    0.963782408119    0.612819896119"%(str(n1+i+1),str(n2+i+3)))
            f.write("\n ")
            f.write("\n     %2s   L     %3s             0.8883464   -0.298398604487    0.348047191182"%(str(n1+i+2),str(n2+i+4)))
            f.write("\n     %2s   L     %3s             0.3047828    1.227982887359    0.722252322062"%(str(n1+i+2),str(n2+i+5)))
            n1+=2
            n2+=5
        f.write("\n ")  
    for imo in range(hf_orbs.shape[0]):
        f.write("\n ")
        f.write("\n          ------------")
        f.write("\n          EIGENVECTORS")
        f.write("\n          ------------")
        f.write("\n ")
        f.write("\n                      %s    "%str(imo+1))
        f.write("\n                   %4f "%(orb_energy[imo]-orb_energy[int((nelec-1)/2)]))
        f.write("\n                     A     ")# symmetry (A is default for c1)
        kao=1
        for jatom, atom in enumerate(atoms_array):
            if atom[0]=='C':
                if file=='allyl' or file=='benzyl':
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                    #f.write("\n  %3s  C %2s  X    %6f"%(str(kao+2),str(jatom+1),hf_orbs[jatom,imo]))
                    #f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    #f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Y    %6f"%(str(kao+3),str(jatom+1),hf_orbs[jatom,imo]))
                    #f.write("\n  %3s  C %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                    f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    kao+=5
                elif file=='dpm' or file=='dpxm' or file=='pdxm':
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    %6f"%(str(kao+2),str(jatom+1),hf_orbs[jatom,imo]))
                    f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                    kao+=5
                else:
                    f.write("\n  %3s  C %2s  S    0.000000  "%(str(kao),str(jatom+1)))
                    f.write("\n  %3s  C %2s  S    0.000000"  %(str(kao+1),str(jatom+1)))
                    f.write("\n  %3s  C %2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                    f.write("\n  %3s  C %2s  Z    %6f"%(str(kao+4),str(jatom+1),hf_orbs[jatom,imo]))
                    kao+=5
            if atom[0]=='Cl':
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao+1),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  X    0.000000  "%(str(kao+2),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Y    0.000000  "%(str(kao+3),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Z    0.000000  "%(str(kao+4),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  S    0.000000  "%(str(kao+5),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  X    0.000000  "%(str(kao+6),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Y    0.000000  "%(str(kao+7),str(jatom+1)))
                f.write("\n  %3s  Cl%2s  Z    %6f"%(str(kao+8),str(jatom+1),hf_orbs[jatom,imo]))
                kao+=9
        f.write("\n  ...... END OF ROHF CALCULATION ......")
    f.write("\n ")
    f.close()
    # check that fock matrix is diagonalized
    fock_mo = np.dot(hf_orbs.T,np.dot(fock_mat,hf_orbs))
    for i in range(fock_mo.shape[0]):
        for j in range(fock_mo.shape[0]):
            if i!=j and fock_mo[i,j] > 1e-4:
                print("Fock matrix not converged!")
                print("\nFock Matrix:")
                print(fock_mo)
                sys.exit()
    # check the density matrix
    dens_mat=mscf.density(hf_orbs,natoms,ndocc)
    dens_mo = np.dot(hf_orbs.T,np.dot(dens_mat,hf_orbs))
    print('\nOrbital occupation numbers:')
    for i in range(dens_mo.shape[0]):
        print("%d: %f"%(i+1,dens_mo[i,i]))
        for j in range(dens_mo.shape[0]):
            if i!=j and fock_mo[i,j] > 1e-4:
                print("Density matrix not converged!")
                print("\nDensity Matrix:")
                print(dens_mo)
                sys.exit()
    if sum(n_list)==0 and natoms_cl==0:
        #strng,ci_energies_array,osc_array = cis(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs,'cis')  
        #strng,ci_energies_array,osc_array = cis(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs,'cisd')  
        strng,ci_energies_array,osc_array,s2_array= cisd.cisd_rot(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs)
    else:
        strng,ci_energies_array,osc_array,s2_array = cisd.hetero_cisd_rot(ndocc,natoms,coord,atoms_array,energy0,two_body,orb_energy,hf_orbs)
    return strng,ci_energies_array,osc_array,s2_array  #return gnuplot data for plotting spectrum

