import math
from binomial import *
from scipy import optimize

def rootBracketing(f, a, b, maxIter, factor):
    for k in range(maxIter):
        if f(a) * f(b) < 0:
            return (a, b)
        if abs(f(a)) < abs(f(b)):
            a += factor * (a-b)  # if f(a) is closer to 0, change a
        else:
            b += factor * (b-a)  # if f(b) is closer to 0, change b
    return (a, b)

def testRootBracketin():
    foo = lambda x : math.exp(x) - 5
    a = 3.4
    b = 5.78
    (a_, b_) = rootBracketing(foo, a, b, 50, 1.6)
    print(a_, b_)

def bisect(f, a, b, tol):
    assert(a < b and f(a) * f(b) < 0)
    c = (a+b) / 2
    while (b-a)/2 > tol:
        print("(a, b) = (", a, ",", b, ")")
        c = (a+b)/2
        if abs(f(c)) < tol:
            return c
        else:
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
    return c

def testBisection():
    # bs price for 10% vol
    price = bsPrice(S=100, r=0.02, vol=0.1, T=1.0, strike=90, payoffType=PayoffType.Call)
    f = lambda vol: (bsPrice(100, 0.02, vol, 1.0, 90, PayoffType.Call) - price)
    a, b = 0.0001, 0.5
    iv = bisect(f, a, b, 1e-6)
    print("implied vol = ", iv)

def secant(f, a, b, tol, maxIter):
    nIter = 0
    c = (a * f(b) - b * f(a)) / (f(b) - f(a))
    while abs(a - b) > tol and nIter <= maxIter:
        print("(a,b) = (", a, ",", b, ")")
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        else:
            a = b
            b = c
        nIter = nIter+1
    return c

def testSecant():
    # bs price for 10% vol
    price = bsPrice(S=100, r=0.02, vol=0.1, T=1.0, strike=90, payoffType=PayoffType.Call)
    f = lambda vol: (bsPrice(100, 0.02, vol, 1.0, 90, PayoffType.Call) - price)
    a, b = 0.0001, 0.5
    iv = secant(f, a, b, 1e-6, 100)
    print("implied vol = ", iv)


def falsi(f, a, b, tol):
    assert (a<b and f(a)*f(b)<0)
    c = (a*f(b)-b*f(a))/(f(b)-f(a))
    while abs(a - b) > tol:
        c = (a*f(b)-b*f(a))/(f(b)-f(a))
        if abs(f(c)) < tol:
            return c;
        else:
            if f(a)*f(c)<0:
                b = c
            else:
                a = c
    return c

def testBrent():
    price = bsPrice(S=100, r=0.02, vol=0.1, T=1.0, strike=90, payoffType=PayoffType.Call)
    f = lambda vol: (bsPrice(100, 0.02, vol, 1.0, 90, PayoffType.Call) - price)
    a, b = 0.0001, 0.5
    iv = optimize.brentq(f, a, b)
    print("implied vol = ", iv)

testBrent()