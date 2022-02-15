import math

from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import bisect
from scipy import optimize

from binomial import PayoffType, cnorm

class Smile:
    def __init__(self, strikes, vols):
        # add additional point on the right to avoid arbitrage
        ks = [strikes[0] - (strikes[1]-strikes[0])/2] + strikes + [1.5 * strikes[-1] - 0.5 * strikes[-2]]
        vs = [vols[0] - 1.0 / 2 * (vols[1] - vols[0])] + vols + [vols[-1] + (vols[-1] - vols[-2]) / 2]

        yp1, ypn = 0.0, 0.0 # end y' are zero, flat extrapolation
        n = len(ks)
        y2 = [0] * n
        u = [0] * (n-1)
        y2[0] = -0.5
        u[0] = (3.0/(ks[1]-ks[0])) * ((vs[1]-vs[0])/(ks[1]-ks[0]) - yp1)
        for i in range(1, n-1):
            sig = (ks[i] - ks[i-1]) / (ks[i+1] - ks[i-1])
            p = sig * y2[i-1] + 2.0
            y2[i] = (sig - 1.0) / p
            u[i] = (vs[i+1] - vs[i]) / (ks[i+1] - ks[i]) - (vs[i]-vs[i-1]) / (ks[i] -ks[i-1])
            u[i] = (6.0 * u[i]/(ks[i+1]-ks[i-1]) - sig*u[i-1]) / p

        qn = 0.5
        un = (3.0/(ks[n-1]-ks[n-2])) * (ypn - (vs[n-1]-vs[n-2]) / (ks[n-1] - (ks[n-2])))
        y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2]+1.0)
        for i in range(n-2, -1, -1):
            y2[i] = y2[i] * y2[i+1] + u[i]

        self.strikes = ks
        self.vols = vs
        self.y2 = y2


        # xs = np.arange(self.strikes[0]/1.1, self.strikes[-1]*1.1, (self.strikes[-1]*1.1 - self.strikes[0]/1.1) / 100)
        # ys = [self.Vol(k) for k in xs]
        # plt.plot(xs, ys, label='smile')
        #
        # plt.plot(self.strikes, self.vols, 'o', color='black');
        # plt.show()

    def Vol(self, k):
        if k < self.strikes[0]:  # scipy cubicspline bc_type confusing, extrapolate by ourselfs
            return self.vols[0]
        if k > self.strikes[-1]:
            return self.vols[-1]
        else:
            idx = 0
            for i in range(len(self.strikes)):
                if k < self.strikes[i]:
                    idx = i
                    break

            h = self.strikes[idx] - self.strikes[idx-1]
            a = (self.strikes[idx] - k) / h
            b = 1-a
            c = (a*a*a - a) * h * h / 6.0
            d = (b*b*b - b) * h * h / 6.0
            return a * self.vols[idx-1] + b*self.vols[idx] + c*self.y2[idx-1] + d*self.y2[idx]

class SmileOld:
    def __init__(self, strikes, vols):
        # add additional point on the right to avoid arbitrage
        self.strikes = strikes  + [1.2*strikes[-1] - 0.2*strikes[-2]]
        self.vols = vols + [vols[-1] + (vols[-1]-vols[-2])/8]
        print(self.strikes, self.vols)
        self.cs = CubicSpline(strikes, vols, bc_type=((1, 0.0), (1, 0.0)), extrapolate=True)

        # xs = np.arange(self.strikes[0], self.strikes[-1], 0.02)
        # ys = [self.cs(k) for k in xs]
        # plt.plot(xs, ys, label='smile')
        # plt.show()

    def Vol(self, k):
        if k < self.strikes[0]:  # scipy cubicspline bc_type confusing, extrapolate by ourselfs
            return self.vols[0]
        if k > self.strikes[-1]:
            return self.vols[-1]
        else:
            return self.cs(k)

class ImpliedVol:
    def __init__(self, ts, smiles):
        self.ts = ts
        self.smiles = smiles
    # linear interpolation in variance, along the strike line
    def Vol(self, t, k):
        # locate the interval t is in
        pos = bisect.bisect_left(self.ts, t)
        # if t is on or in front of first pillar,
        if pos == 0:
            return self.smiles[0].Vol(k)
        if pos >= len(self.ts):
            return self.smiles[-1].Vol(k)
        else:  # in between two brackets
            prevVol, prevT = self.smiles[pos-1].Vol(k), self.ts[pos-1]
            nextVol, nextT = self.smiles[pos].Vol(k), self.ts[pos]
            w = (nextT - t) / (nextT - prevT)
            prevVar = prevVol * prevVol * prevT
            nextVar = nextVol * nextVol * nextT
            return  math.sqrt((w * prevVar + (1-w) * nextVar)/t)
        return

    def dVoldK(self, t, k):
        return (self.Vol(t, k+0.01) - self.Vol(t, k-0.01)) / 0.02
    def dVoldT(self, t, k):
        return (self.Vol(t+0.005, k) - self.Vol(t, k)) / 0.005
    def dVol2dK2(self, t, k):
        return (self.Vol(t, k+0.01) + self.Vol(t, k-0.01) - 2*self.Vol(t, k)) / 0.0001


class LocalVol:
    def __init__(self, iv, S0, rd, rf):
        self.iv = iv
        self.S0 = S0
        self.rd = rd
        self.rf = rf
    def LV(self, t, s):
        if t < 1e-6:
            return self.iv.Vol(t, s)
        imp = self.iv.Vol(t, s)
        dvdk = self.iv.dVoldK(t, s)
        dvdt = self.iv.dVoldT(t, s)
        d2vdk2 = self.iv.dVol2dK2(t, s)
        d1 = (math.log(self.S0/s) + (self.rd-self.rf)*t + imp * imp * t / 2) / imp / math.sqrt(t)
        numerator = imp*imp + 2*t*imp*dvdt + 2*(self.rd-self.rf)*s*t*imp*dvdk
        denominator = (1+s*d1*math.sqrt(t)*dvdk)**2 + s*s*t*imp*(d2vdk2 - d1 * math.sqrt(t) * dvdk * dvdk)
        localvar = min(max(numerator / denominator, 1e-8), 1.0)
        if numerator < 0: # floor local volatility
            localvar = 1e-8
        if denominator < 0: # cap local volatility
            localvar = 1.0
        return math.sqrt(localvar)

def testSmile():
    smile1W = Smile([0.3, 0.8, 1.0, 1.2, 1.5], [0.15, 0.12, 0.1, 0.14, 0.17])
    xs = np.arange(0.2, 1.8, 0.02)
    ys = [smile1W.Vol(k) for k in xs]
    plt.plot(xs, ys, label='smile')
    plt.show()


def fwdDelta(fwd, stdev, strike, payoffType):
    d1 = math.log(fwd / strike) / stdev + stdev / 2
    if payoffType == PayoffType.Call:
        return cnorm(d1)
    elif payoffType == PayoffType.Put:
        return - cnorm(-d1)
    else:
        raise Exception("not supported payoff type", payoffType)

# solve for the K such that Delta(S, T, K, vol) = delta
def strikeFromDelta(S, r, q, T, vol, delta, payoffType):
    fwd = S * math.exp((r-q) * T)
    if payoffType == PayoffType.Put:
        delta = -delta
    f = lambda K: (fwdDelta(fwd, vol * math.sqrt(T), K, payoffType) - delta)
    a, b = 0.0001, 10000
    return optimize.brentq(f, a, b)


def smileFromMarks(T, S, r, q, atmvol, bf25, rr25, bf10, rr10):
    c25 = bf25 + atmvol + rr25/2
    p25 = bf25 + atmvol - rr25/2
    c10 = bf10 + atmvol + rr10/2
    p10 = bf10 + atmvol - rr10/2

    ks = [ strikeFromDelta(S, r, q, T, p10, 0.1, PayoffType.Put),
           strikeFromDelta(S, r, q, T, p25, 0.25, PayoffType.Put),
           S * math.exp((r-q)*T),
           strikeFromDelta(S, r, q, T, c25, 0.25, PayoffType.Call),
           strikeFromDelta(S, r, q, T, c10, 0.1, PayoffType.Call) ]
    # print(T, ks)
    return SmileOld(ks, [p10, p25, atmvol, c25, c10])


def createTestImpliedVol(S, r, q, sc):
    pillars = [0.02, 0.04, 0.06, 0.08, 0.16, 0.25, 0.75, 1.0, 1.5, 2, 3, 5]
    atmvols = [0.155, 0.1395, 0.1304, 0.1280, 0.1230, 0.1230, 0.1265, 0.1290, 0.1313, 0.1318, 0.1313, 0.1305, 0.1295]
    bf25s = [0.0016, 0.0016, 0.0021, 0.0028, 0.0034, 0.0043, 0.0055, 0.0058, 0.0060, 0.0055, 0.0054, 0.0050, 0.0045, 0.0043]
    rr25s = [-0.0065, -0.0110, -0.0143, -0.0180, -0.0238, -0.0288, -0.0331, -0.0344, -0.0349, -0.0340, -0.0335, -0.0330, -0.0330]
    bf10s = [0.0050, 0.0050, 0.0067, 0.0088, 0.0111, 0.0144, 0.0190, 0.0201, 0.0204, 0.0190, 0.0186, 0.0172, 0.0155, 0.0148]
    rr10s = [-0.0111, -0.0187, -0.0248, -0.0315, -0.0439, -0.0518, -0.0627, -0.0652, -0.0662, -0.0646, -0.0636, -0.0627, -0.0627]
    smiles = [smileFromMarks(pillars[i], S, r, q, atmvols[i], bf25s[i]*sc, rr25s[i]*sc, bf10s[i]*sc, rr10s[i]*sc) for i in range(len(pillars))]
    return ImpliedVol(pillars, smiles)


def plotTestImpliedVolSurface():
    S, r, q = 1.25805, 0.01, 0.003
    iv = createTestImpliedVol(S, r, q, 0.5)
    tStart, tEnd = 0.02, 1
    ts = np.arange(tStart, tEnd, 0.1)
    fwdEnd = S*math.exp((r-q)*tEnd)
    kmin = strikeFromDelta(S, r, q, tEnd, iv.Vol(tEnd, fwdEnd), 0.05, PayoffType.Put)
    kmax = strikeFromDelta(S, r, q, tEnd, iv.Vol(tEnd, fwdEnd), 0.05, PayoffType.Call)
    ks = np.arange(kmin, kmax, 0.01)

    vs = np.ndarray((len(ts), len(ks)))
    for i in range(len(ts)):
        for j in range(len(ks)):
            vs[i, j] = iv.Vol(ts[i], ks[j])
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(ks, ts)
    ha.plot_surface(X, Y, vs)
    plt.show()

# testSmile()
# plotTestImpliedVolSurface()