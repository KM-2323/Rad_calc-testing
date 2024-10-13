def write_gnu(strng,file, jobname):
    f=open('gnuplot_script_%s_%s'%(jobname,file),'w')
    f.write("#simulated spectrum\n")
    f.write("set term pdf size 6,4\n")
    f.write("unset key\n")
    f.write("set output '%s.pdf'\n" %(file))
    f.write("set xrange [200:700]\n")
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm' font ',18'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units' font ',18'\n")
    f.write("set xtics font ',18'\n")
    f.write("set ytics font ',18'\n")
    f.write("set bmargin 4\n")
    f.write("p %s lw 3 dt 1" %strng)
    f.close()
    return
    


def multi_gnu(figure,rnge):
    figurename=""
    for molecule in figure:
        figurename += molecule +'_'
    figurename = figurename[:-1]
    f=open('gnuplot_script_'+figurename,'w')
    f.write("# ROPPP simulated spectrum\n")
    f.write("set output '%s.pdf'\n" %(figurename))
    f.write("set term pdf size 6,8\n")
    f.write("set multiplot\n")
    f.write("set key right top font ',16'\n")
    f.write("set xrange %s\n"%rnge)
    f.write("set samples 10000\n")
    f.write("set xlabel 'Wavelength / nm' font ',16'\n")
    f.write("set ylabel 'Absorbance / Arbitrary Units' font ',16'\n")
    f.write("set xtics font ',14'\n")
    f.write("set ytics font ',14'\n")
    f.write("set size 0.9, 0.45\n")
    f.write("set lmargin 10\n")
    f.write("set rmargin 0\n")
    f.write("set tmargin 0\n")
    f.write("set bmargin 5\n")
    f.write("set title 'ROPPP'\n")
    f.write("p")
    f.close()
    return figurename


def write_comp(D1_all, Bright_all, jobname):
    # Write D1 energies to file
    with open(f"compD1_{jobname}.txt", "w") as d1_file:
        for entry in D1_all:
            molecule, d1_exp, d1_calc = entry
            d1_file.write(f'{molecule}\n')
            d1_file.write(f'[{d1_exp}, {d1_calc}]\n')
    
    # Write Bright energies to file
    with open(f"compBrght_{jobname}.txt", "w") as bright_file:
        for entry in Bright_all:
            if len(entry) >= 4:
                molecule, bright_exp1, bright_calc1, bright_exp2, bright_calc2 = entry
                bright_file.write(f'{molecule}\n')
                bright_file.write(f'[{bright_exp1}, {bright_calc1},{bright_exp2}, {bright_calc2} ]\n')
            else:
                molecule, bright_exp, bright_calc= entry
                bright_file.write(f'{molecule}\n')
                bright_file.write(f'[{bright_exp}, {bright_calc}]\n')

