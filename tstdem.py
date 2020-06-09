import mcdem

fds = [('tspec/dsp1.txt',550.0,580.0,1)]
ftg = 'tspec/tg.txt'
fdg = 'tspec/dg.txt'
fcg = 'tspec/cg.txt'
fsg = 'tspec/s%dt%02dc%02dd%02d.pt'
ips = [(0.0,0.0,0.0),
       (10.0,9.0,11.5),
       (1.0,1e-3,1e3),
       (0.0,-2.0,2.0),
       (0.0,-0.1,0.1),
       (0.5,0.01,5.0)]
d = mcdem.run_mcdem(fds, ftg, fdg, fcg, fsg, ips=ips,
                    sav=['O02dem.pkl', 500], imp=3000, npr=50)
