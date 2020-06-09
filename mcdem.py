from pylab import *
from scipy import integrate, interpolate, special, stats, linalg
import pickle
import time, os
from collections import namedtuple

SpecType = namedtuple('SpecType',
                      ['xlo', 'xhi', 'xm', 'yd', 'eff', 'lnyd',
                       'imax', 'xmax', 'yb', 'ym', 'nps'])

def transpec(xm, y, w, im, a, b):
    nx = len(xm)
    if w > 0:
        nm = int(nx/2)
        x = xm-xm[nm]
        r = stats.norm(scale=w).pdf(x)
        r /= sum(r)
        y = real(ifft(fft(y)*conjugate(fft(r))))        
        y = append(y[nm:],y[:nm])
    if a != 0 or b != 0:
        xi = xm[im]+(1+b)*(xm-xm[im])+a
        yi = cumsum(y)
        fiy = interpolate.interp1d(xi, yi, bounds_error=False,
                                   fill_value=(yi[0],yi[-1]))
        dx = (xm[1]-xm[0])
        xx = append(xm-0.5*dx,xm[-1]+dx)        
        yx = fiy(xx)
        y = yx[1:]-yx[:-1]
    return y
    
def read_spec(fn, emin, emax, yb=1e-5, feff=''):
    r = loadtxt(fn, unpack=1)
    w = where((r[0]>=emin)&(r[0]<=emax))
    w = w[0]
    xlo = r[0][w]
    xhi = append(xlo[1:],2*xlo[-1]-xlo[-2])
    dx = xhi[0]-xlo[0]
    xlo -= 0.5*dx
    xhi -= 0.5*dx
    yd = r[1][w] + yb
    nx = len(yd)
    eff = zeros(nx)
    eff[:] = 1.0
    xm = 0.5*(xlo+xhi)
    if feff != '':
        e = loadtxt(feff, unpack=1)
        fie = interpolate.interp1d(e[0], e[1], bounds_error=False,
                                   fill_value='extrapolate')
        eff = fie(xm)
        
    lnyd = special.gammaln(yd+1.0)-log(sqrt(2*pi*yd))
    imax = argmax(yd)
    xmax = xm[imax]
    s = SpecType(xlo, xhi, xm, yd, eff, lnyd, imax, xmax,
                 [yb,1.0,0.0,0.0],
                 ['ymt', 'ymi', 'ymb', 'xb', 'ymg', 'yms'],
                 ['ib', 'nb', 'ic', 'id', 'ia']+([0]*5))
    return s

def setup_model(d, fp):
    tg = d['tg']
    cg = d['cg']
    dg = d['dg']
    nt = len(tg)
    nc = len(cg)
    nd = len(dg)
    s = d['sp']
    ns = len(s)    
    for k in range(ns):
        nx = len(s[k].yd)
        s[k].ym[4] = zeros((nt,nc,nd,nx))
        for it in range(nt):
            for ic in range(nc):
                for id in range(nd):
                    fn = fp%(k,it,ic,id)
                    r = loadtxt(fn, unpack=1)
                    fir = interpolate.interp1d(r[0],r[1],
                                               bounds_error=False,
                                               fill_value=0.0)
                    s[k].ym[4][it,ic,id] = fir(s[k].xm)*s[k].eff
                    
def lnlikely(d, ip):
    s = d['sp']
    ns = len(s)
    ifx = d['ifx']
    cc = d['cc']
    lnlk = d['lnlk']
    p = d['mp']
    tg = d['tg']
    cg = d['cg']
    dg = d['dg']
    ik = d['ik']
    nt = len(tg)
    nc = len(cg)
    nd = len(dg)
    dc = 0.0
    dd = 0.0
    if len(cg) > 1:
        dc = cg[1]-cg[0]
    if len(dg) > 1:
        dd = dg[1]-dg[0]
    
    def update_yb(k):
        ib = s[k].nps[0]
        nb = s[k].nps[1]
        ie = ib+nb
        if nb == 1:
            s[k].ym[2] = repeat(p[ib], len(s[k].xlo))
        else:
            fib = interpolate.interp1d(s[k].sym[3], p[ib:ie],
                                       bounds_error=False,
                                       fill_value='extrapolate')
            s[k].ym[2] = fib(s[k].xm)

    def update_ym1(k):
        ic = s[k].nps[2]
        id = s[k].nps[3]
        pc = p[ic]
        pd = p[id]
        ic0 = -1
        id0 = -1
        if dc > 0:
            ic0 = int((pc-cg[0])/dc)
        if dd > 0:
            id0 = int((pd-dg[0])/dd)
        if ic0 < 0:
            ic0 = 0
            ic1 = 0
        elif ic0 >= nc-1:
            ic0 = nc-1
            ic1 = nc-1
        else:
            ic1 = ic0+1
        if id0 < 0:
            id0 = 0
            id1 = 0
        elif id0 >= nd-1:
            id0 = nd-1
            id1 = nd-1
        else:
            id1 = id0+1
        fc = 0.0
        fd = 0.0
        if ic1 > ic0:
            fc = (pc-cg[ic0])/dc
        if id1 > id0:
            fd = (pd-dg[id0])/dd
        yg = s[k].ym[4]
        for t in range(nt):
            y00 = yg[t,ic0,id0]
            y01 = yg[t,ic0,id1]
            y10 = yg[t,ic1,id0]
            y11 = yg[t,ic1,id1]
            if fd == 0:
                ym0 = y00
                ym1 = y10
            elif fd == 1:
                ym0 = y01
                ym1 = y11
            else:
                if fc == 0:
                    ym0 = y00*(1-fd)+y01*fd
                    ym1 = ym0
                elif fc == 1:
                    ym1 = y10*(1-fd)+y11*fd
                    ym0 = ym1
                else:
                    ym0 = y00*(1-fd)+y01*fd
                    ym1 = y10*(1-fd)+y11*fd
            if fc == 0:
                s[k].ym[1][t] = ym0
            elif fc == 1:
                s[k].ym[1][t] = ym1
            else:
                s[k].ym[1][t] = ym0*(1-fc)+ym1*fc

    def update_ym0(k):
        s[k].ym[0] = matmul(p[:nt], s[k].ym[1])
        i1 = s[k].nps[5]
        i2 = s[k].nps[6]
        i3 = s[k].nps[7]
        a = p[i1]
        b = p[i2]
        w = p[i3]
        if w > 0 or a != 0 or b != 0:
            s[k].ym[0] = transpec(s[k].xm, s[k].ym[0], w, s[k].imax, a, b)
        s[k].yb[2] = sum(s[k].ym[0])

    def update_ysc():
        yt1 = 0.0
        yt0 = 0.0
        for k in range(ns):
            yt1 += s[k].yb[2]
            if s[k].yb[3] == 0.0:
                s[k].yb[3] = s[k].yb[2]
            yt0 += s[k].yb[3]
        for k in range(ns):
            s[k].yb[1] = yt0/yt1

    def update_lnlk(k):
        ia = s[k].nps[4]
        s[k].ym[5] = s[k].eff*s[k].ym[0]*s[k].yb[1]*p[ia] + s[k].ym[2]
        ym = s[k].ym[5]+s[k].yb[0]
        lnlk[k] = sum(s[k].yd*log(ym)-ym-s[k].lnyd)
        
    if ip >= 0:
        w = (where(ifx == ip))[0]
        iu = append(w, ip)
        for i in iu:
            if i != ip:
                uc = cc[i]
                xp = p[ip]
                if uc[2]:
                    xp = xp**uc[2]
                if uc[0]:
                    xp = uc[0]*xp
                if uc[1]:
                    xp += uc[1]
                if uc[3]:
                    xp = xp**uc[3]
                p[i] = xp
            if i < nt:
                for k in range(ns):
                    update_ym0(k)
                update_ysc()
                for k in range(ns):
                    update_lnlk(k)
                continue
            
            k = ik[i]
            ib = s[k].nps[0]
            ib1 = ib + s[k].nps[1]
            if i >= ib and i < ib1:
                update_yb(k)
                update_lnlk(k)
                continue            
            if i == s[k].nps[4]:
                update_lnlk(k)
                continue
            update_ym1(k)
            update_ym0(k)
            update_ysc()
            update_lnlk(k)
            continue
    else:
        for k in range(ns):
            update_ym1(k)
            update_ym0(k)
        update_ysc()
        for k in range(ns):
            update_yb(k)
            update_lnlk(k)
            
    return sum(lnlk)

def logqx(x):
    x = abs(x)
    b1 =  0.319381530
    b2 = -0.356563782
    b3 =  1.781477937
    b4 = -1.821255978
    b5 =  1.330274429
    t = 1.0/(1+0.2316419*x)
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t
    zx = -0.5*x*x + log(b1*t + b2*t2 + b3*t3 + b4*t4 + b5*t5) - 0.918938533205
    return zx

def fit_mcdem(d, imp, sav=[], racc=0.4, nburn=0, npr=1):
    xin = arange(-30.0, 0.05, 0.05)
    xin[-1] = 0.0
    yin = integrate.cumtrapz(stats.norm().pdf(xin), xin, initial=0.0)
    yin[0] = yin[1]*(yin[1]/yin[2])
    fin0 = interpolate.interp1d(xin, yin, kind='linear',
                                bounds_error=False, fill_value=(yin[0],0.5))
    fin1 = interpolate.interp1d(log(yin), xin, kind='linear',
                                bounds_error=False, fill_value=(-30.0,0.0))
    def cnorm(x):
        if (x > 0):
            if x > 30:
                z = logqx(x)
                if z < -300:
                    return 1.0
                else:
                    return 1-exp(z)
            return 1-fin0(-x)
        elif (x < 0):
            if x < -30:
                z = logqx(x)
                if z < -300:
                    return 0.0
                else:
                    return exp(z)
            return fin0(x)
        else:
            return 0.5
        
    def inorm(y):
        if y > 0.5:            
            return -fin1(log(1-y))
        elif y < 0.5:
            return fin1(log(y))
        else:
            return 0.0
           
    def rand_cg(x0, x1):
        r = rand()
        r0 = cnorm(x0)
        r1 = cnorm(x1)
        r = r0 + r*(r1-r0)
        return (inorm(r),r,r0,r1)

    sp = d['sp']
    ns = len(sp)
    mp0 = d['mp0']
    mp1 = d['mp1']
    smp = d['smp']
    mp = d['mp']
    np = len(mp)
    ifx = d['ifx']
    cc = d['cc']
    tg = d['tg']
    nt = len(tg)
    
    def update_dem(r0, sar, rar, har):
        trej = 0.0
        nrej = 0
        for ip in range(nt):
            for jp in range(ip):
                imin = mp0[ip]
                imax = mp1[ip]
                mpi = mp[ip]
                jmin = mp0[jp]
                jmax = mp1[jp]
                mpj = mp[jp]
                sigma = sar[ip,jp]
                xmax = min(imax-mpi, mpj-jmin)
                xmin = max(imin-mpi, mpj-jmax)
                xp0 = xmin/sigma
                xp1 = xmax/sigma
                (rn,yp,y0,y1) = rand_cg(xp0, xp1)
                dp = sigma*rn
                mp[ip] += dp
                mp[jp] -= dp
                xmaxi = min(imax-mp[ip], mp[jp]-jmin)
                xmini = max(imin-mp[ip], mp[jp]-jmax)
                xpi0 = xmini/sigma
                xpi1 = xmaxi/sigma
                yi0 = cnorm(xpi0)
                yi1 = cnorm(xpi1)
                r = lnlikely(d, ip)
                dr = r + log(y1-y0)-log(yi1-yi0)
                rej = 0
                rp = 0.0
                if dr < r0:
                    rp = 1-exp(dr-r0)
                    if rand() < rp:                        
                        mp[ip] = mpi
                        mp[jp] = mpj
                        r = lnlikely(d, ip)
                        rej = 1
                if not rej:
                    r0 = r
                    har[i,ip,jp] = dp
                trej += rp
                nrej += 1
                rar[i,ip,jp] = rp

        return (r0,trej,nrej)

    t0 = time.time()
    r0 = lnlikely(d, -1)
    frej = zeros((imp,np),dtype=int8)
    rrej = zeros((imp,np))
    rrej[:,:] = -1.0
    arej = zeros((imp,nt,nt))
    arej[:,:,:] = -1.0
    sda = zeros((nt,nt))
    hda = zeros((imp,nt,nt))
    fda = zeros((nt,nt))
    fmp = zeros(np)
    fmp[:] = 1.0
    fda[:,:] = 1.0
    for ip in range(nt):
        for jp in range(nt):
            sda[ip,jp] = sqrt(smp[ip]*smp[jp])
    hmp = zeros((imp,np))
    ene = zeros(imp)
    hmp[0] = mp
    trej = 0.0
    ene[0] = r0
    if nburn <= 0:
        nburn = 0.2
    if nburn < 1:
        nburn = int32(nburn*imp)
    ttr = 0.0
    
    if len(sav) == 3:
        fsav = sav[0]
        nsav = sav[1]
        tsav = sav[2]
    if len(sav) == 2:
        fsav = sav[0]
        nsav = sav[1]
        tsav = fsav+'.trigger'
    elif len(sav) == 1:
        fsav = sav[0]
        nsav = 0
        tsav = fsav+'.trigger'
    else:
        fsav = None
        nsav = 0
        tsav = None

    for i in range(1, imp):
        i1 = i-1
        trej = 0.0
        nrej = 0.0
        r0,treja,nreja = update_dem(r0, sda, arej, hda)
        nrej += nreja
        trej += treja

        for ip in range(nt,np):
            if ifx[ip] >= 0:
                continue
            if mp1[ip] <= mp0[ip]:
                continue
            xp0 = (mp0[ip]-hmp[i1,ip])/smp[ip]
            xp1 = (mp1[ip]-hmp[i1,ip])/smp[ip]
            rn,yp,y0,y1 = rand_cg(xp0, xp1)
            mp[ip] = hmp[i1,ip] + rn*smp[ip]
            yi0 = cnorm((mp0[ip]-mp[ip])/smp[ip])
            yi1 = cnorm((mp1[ip]-mp[ip])/smp[ip])
            r = lnlikely(d, ip)
            dr = r + log(y1-y0)-log(yi1-yi0)
            rej = 0
            rp = 0.0
            if dr < r0:
                rp = 1-exp(dr-r0)
                if rand() < rp:
                    mp[ip] = hmp[i1,ip]
                    r = lnlikely(d, ip)
                    rej = 1
            frej[i,ip] = rej
            if not rej:
                r0 = r
                
            rrej[i,ip] = rp
            trej += rp
            nrej += 1
            
        hmp[i] = mp
        ene[i] = r0
        if i >= 50 and i <= nburn and i%25 == 0:
            im = i-25
            im0 = max(i-100,10)
            for ip in range(nt):
                for jp in range(ip):
                    fa = mean(arej[im:i+1,ip,jp])
                    fa = fda[ip,jp]*((1-fa)/racc)**2
                    fa = min(fa, 1e2)
                    fa = max(fa, 1e-2)
                    fa = 0.25*fda[ip,jp]+0.75*fa
                    xst = fa*std(hda[im0:i+1,ip,jp])
                    fda[ip,jp] = fa
                    if xst > 0:
                        sda[ip,jp] = xst
            for ip in range(nt,np):
                ra = mean(rrej[im:i+1,ip])
                fa = fmp[ip]*((1-ra)/racc)**2
                fa = min(fa, 1e2)
                fa = max(fa, 1e-2)
                fa = 0.25*fmp[ip]+0.75+fa
                xst = fa*std(hmp[im0:i+1,ip])
                fmp[ip] = fa
                if xst > 0:
                    smp[ip] = xst
        trej /= nrej
        ttr = (ttr*(i-1)+trej)/i
        war = where(arej[i] >= 0)
        wr = where(rrej[i] >= 0)
        arm = 0.0
        rrm = 0.0
        if len(war[0]) > 0:
            arm = mean(arej[i,war[0],war[1]])
        if len(wr[0]) > 0:
            rrm = mean(rrej[i,wr[0]])
        pp = [i,trej,ttr,rrm,arm,r0,r0-ene[i1],time.time()-t0]
        if (i+1)%npr == 0:
            print('imc: %6d %7.1E %7.1E %7.1E %7.1E %12.5E %8.1E %10.4E'%tuple(pp))
        savenow = False
        
        if i == imp-1:
            savenow = True
        elif nsav > 0 and (i+1)%nsav == 0:
            savenow = True
        elif tsav != None and os.path.isfile(tsav):
            savenow = True
            os.system('rm '+tsav)            
        if savenow:
            print('pickling: %s %10.3E'%(fsav, time.time()-t0))
            if nsav > 0:
                with open(fsav,'wb') as fs:
                    d['hmp'] = hmp
                    d['ene'] = ene
                    d['frej'] = frej
                    d['rrej'] = rrej
                    d['arej'] = arej
                    d['sda'] = sda
                    d['fda'] = fda
                    d['fmp'] = fmp
                    d['mpe'] = std(hmp[:imp],0)
                    pickle.dump(d, fs)
    print('done: %10.4E'%(time.time()-t0))

def sim_spec(ifn, ofn='', sa=0.0, sm=1e3, sb=10.0, xa=0.0, xb=0.0, xw=0.0):
    r = loadtxt(ifn, unpack=1)
    yd = r[1]
    if sa > 0:
        yd *= sa
    else:
        sa = sm/max(r[1])
        yd *= sa
    im = argmax(yd)
    yd = transpec(r[0], yd, xw, im, xa, xb)
    yc = poisson(yd+sb)
    if ofn != '':
        savetxt(ofn, transpose((r[0],yc)))
    return r[0],yc,sa

def setup_mcdem(fds, ftg, fdg, fcg, fsg, ips=[]):
    d = {}
    tg = loadtxt(ftg, unpack=1, ndmin=2)[0]
    cg = loadtxt(fcg, unpack=1, ndmin=2)[0]
    dg = loadtxt(fdg, unpack=1, ndmin=2)[0]
    d['tg'] = tg
    d['cg'] = cg
    d['dg'] = dg
    nt = len(tg)
    ns = len(fds)
    sp = []    
    np = nt + ns*6
    nbs = 0
    for fd in fds:
        nbs += fd[3]
    np += nbs
    ip = nt
    mp = zeros(np)
    p0 = zeros(np)
    p1 = zeros(np)
    ps = zeros(np)
    ik = zeros(np, dtype=int32)
    ifx = zeros(np, dtype=int32)
    ifx[:] = -1
    cc = zeros((np,4))
    for i in range(ns):
        fd = fds[i]
        if len(fd)==5:
            fe = fd[4]
        else:
            fe = ''
        s = read_spec(fd[0], fd[1], fd[2], feff=fe)
        nx = len(s.yd)
        s.nps[0] = ip
        s.nps[1] = fd[3]
        ip += fd[3]
        ik[s.nps[0]:ip] = i
        s.nps[2:8] = range(ip,ip+6)
        ik[ip:ip+6] = i
        ip += 6
        if i > 0:
            ifx[s.nps[2]] = sp[0].nps[2]
            ifx[s.nps[3]] = sp[0].nps[3]
        s.ym[0] = zeros(nx)
        s.ym[1] = zeros((nt,nx))
        s.ym[2] = zeros(nx)
        s.ym[5] = zeros(nx)
        if fd[3] > 1:
            e0 = min(s.xlo)
            e1 = max(s.xhi)
            s.ym[3] = arange(e0, e1, (e1-e0)/(fd[3]-1))
        else:
            s.ym[3] = zeros(0)
        w = where(s.yd-s.yb[0] > 0)
        ib0 = s.nps[0]
        ib1 = s.nps[0]+s.nps[1]
        mp[ib0:ib1] = min(s.yd[w])
        p0[ib0:ib1] = mp[ib0:ib1]*1e-3
        p1[ib0:ib1] = mp[ib0:ib1]*1e3
        ps[ib0:ib1] = mp[ib0:ib1]*0.1
        for j in range(len(ips)):
            ix = s.nps[j+2]
            px = ips[j]
            mp[ix] = px[0]
            p0[ix] = px[1]
            p1[ix] = px[2]
            ps[ix] = (px[2]-px[1])*0.05
        sp.append(s)
    d['sp'] = sp
    mp[:nt] = 1.0/nt
    p0[:nt] = 0.0
    p1[:nt] = 1.0
    ps[:nt] = 0.05
    d['mp'] = mp
    d['mp0'] = p0
    d['mp1'] = p1
    d['smp'] = ps
    d['ifx'] = ifx
    d['cc'] = cc
    d['lnlk'] = zeros(ns)
    d['ik'] = ik
        
    setup_model(d, 'tspec/s%dt%02dc%02dd%02d.pt')
    r = lnlikely(d, -1)
    for i in range(ns):
        ia = sp[i].nps[4]
        mp[ia] = sum(sp[i].yd)/sp[i].yb[3]
        p0[ia] = 1e-3*mp[ia]
        p1[ia] = 1e3*mp[ia]
        ps[ia] = 0.1*mp[ia]

    return d

def run_mcdem(fd, ftg, fdg, fcg, fsg, ips=[],
              imp=3000, sav=[], npr=1):
    d = setup_mcdem(fd, ftg, fdg, fcg, fsg, ips=ips)
    fit_mcdem(d, imp, sav=sav, npr=npr)
    return d

def load_pkl(fn):
    with open(fn, 'rb') as f:
        d = pickle.load(f)
        p = d['hmp']
        w = where(d['ene'] < 0)
        w = w[0]
        nw = len(w)
        i0 = min(1000,int(nw/2))
        d['mp'] = mean(p[i0:nw],0)
        r = lnlikely(d, -1)
        d['mpe'] = std(p[i0:nw],0)
    return d

def plot_hp(z, i, bins=25, xsc=0, xlab='', op=0):
    if op == 0:
        clf()
    w = where(z['ene'] < 0)
    w = w[0]
    nw = len(w)
    im = min(1000,int(nw/2))
    y = z['hmp'][im:nw,i]
    h = histogram(y, bins=bins)
    x = h[1][:-1]
    if xsc == 0:
        plot(x, h[0]/max(h[0]), drawstyle='steps')
    elif xsc == 1:
        semilogx(10**x, h[0]/max(h[0]), drawstyle='steps')
    xlabel(xlab)
    ym = mean(y)
    ys = std(y)
    text(ym+1.5*ys, 0.9, '%.2f+/-%0.2f'%(ym,ys))
    return ym, ys

def plot_rspec(d, i, xr=[], sav='', every=1):
    a1 = plt.subplot2grid((4,1),(0,0),rowspan=3)
    a2 = plt.subplot2grid((4,1),(3,0),rowspan=1)
    s = d['sp'][i]
    xm = 0.5*(s.xlo+s.xhi)
    yd = s.yd
    ym = s.eff*(s.ym[5])+s.yb[0]
    ye = sqrt(s.yd+1.0)
    if every > 1:
        xm = xm[::every]
        ym = ym[::every]
        ye = ye[::every]
        yd = yd[::every]
    a1.errorbar(xm, yd, yerr=ye, marker='.', capsize=3, fmt=' ')
    a1.plot(xm, ym)
    a2.errorbar(xm, (yd-ym)/ye, yerr=1.0, marker='.', capsize=3, fmt=' ')
    a2.plot([min(xm),max(xm)],[0.0,0.0])
    a2.set_ylim(-4,4)
    a2.set_xlabel('Energy (eV)')
    a1.set_ylabel('Counts')
    a2.set_ylabel(r'$\chi^2$')
    a1.set_xticklabels([])
    if len(xr) == 2:
        a1.set_xlim(xr[0], xr[1])
        a2.set_xlim(xr[0], xr[1])
    if sav != '':
        savefig(sav)

def plot_dem(d, op=0, sav='', ylog=0):
    if not op:
        clf()
    
    tg = d['tg']
    p = d['mp']
    dp = d['mpe']
    nt = len(tg)
    errorbar(tg, p[:nt], yerr=dp[:nt], marker='o', capsize=3,
             drawstyle='steps-mid')
    xlabel('Log[T (eV)]')
    ylabel('DEM')
    if ylog:
        yscale('log')
    if sav != '':
        savefig(sav)
        
