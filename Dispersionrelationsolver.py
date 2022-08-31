import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import fsolve
from sympy import integrate, Symbol
from sympy.abc import x
import matplotlib.pyplot as plt
def func1re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(3))
def func1im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(3))
def func2re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(3))
def func2im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(3))
def func3re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(-3))
def func3im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(-3))
def func5re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(-3))
def func5im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(-3))
def func6re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*np.sqrt(1-x**2)*(3))
def func6im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*np.sqrt(1-x**2)*(3))
def func7re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*np.sqrt(1-x**2)*(-3))
def func7im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(-3)*np.sqrt(1-x**2))
def func8re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(3)*np.sqrt(1-x**2))
def func8im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(3)*np.sqrt(1-x**2))
def func9re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(-3)*np.sqrt(1-x**2))
def func9im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(-3)*np.sqrt(1-x**2))
def func10re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(3)*x)
def func10im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(3)*x)
def func11re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(3)*x)
def func11im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(3)*x)
def func12re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(3)*x)
def func12im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(3)*x)
def func13re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(3)*x)
def func13im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(3)*x)
def func14re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(1-x**2)*(3))
def func14im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(3)*(1-x**2))
def func15re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(3)*(1-x**2))
def func15im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(3)*(1-x**2))
def func16re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(-3)*(1-x**2))
def func16im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(-3)*(1-x**2))
def func17re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(-3)*(1-x**2))
def func17im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(-3)*(1-x**2))
def func18re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*x**2*(3))
def func18im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*(3)*x**2)
def func19re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim -b*x + c*np.sqrt(1 - x**2)))*(3)*x**2)
def func19im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim -b*x + c*np.sqrt(1 - x**2)))*(3)*x**2)
def func20re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim +b*x + c*np.sqrt(1 - x**2)))*(-3)*x**2)
def func20im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim +b*x + c*np.sqrt(1 - x**2)))*(-3)*x**2)
def func21re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim +b*x - c*np.sqrt(1 - x**2)))*(-3)*x**2)
def func21im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim +b*x - c*np.sqrt(1 - x**2)))*(-3)*x**2)
def func22re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*np.sqrt(1-x**2)*(3)*x)
def func22im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x - c*np.sqrt(1 - x**2)))*np.sqrt(1-x**2)*(3)*x)
def func23re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*np.sqrt(1-x**2)*(-3)*x)
def func23im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim - b*x + c*np.sqrt(1 - x**2)))*(-3)*np.sqrt(1-x**2)*x)
def func24re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(-3)*np.sqrt(1-x**2)*x)
def func24im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x + c*np.sqrt(1 - x**2)))*(-3)*np.sqrt(1-x**2)*x)
def func25re(x, wre, wim, b, c):
    return np.real((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(3)*np.sqrt(1-x**2)*x)
def func25im(x, wre, wim, b, c):
    return np.imag((1/(wre + 1j*wim + b*x - c*np.sqrt(1 - x**2)))*(3)*np.sqrt(1-x**2)*x)
k1re = lambda wre, wim, b, c: quad(func1re, 0, 1, args=(wre, wim, b, c))[0]
k1im = lambda wre, wim, b, c: quad(func1im, 0, 1, args=(wre, wim, b, c))[0]
k2re = lambda wre, wim, b, c: quad(func2re, 0, 1, args=(wre, wim, b, c))[0]
k2im = lambda wre, wim, b, c: quad(func2im, 0, 1, args=(wre, wim, b, c))[0]
k3re = lambda wre, wim, b, c: quad(func3re, 0, 1, args=(wre, wim, b, c))[0]
k3im = lambda wre, wim, b, c: quad(func3im, 0, 1, args=(wre, wim, b, c))[0]
k4re = lambda wre, wim, b, c: quad(func5re, 0, 1, args=(wre, wim, b, c))[0]
k4im = lambda wre, wim, b, c: quad(func5im, 0, 1, args=(wre, wim, b, c))[0]
k5re = lambda wre, wim, b, c: quad(func6re, 0, 1, args=(wre, wim, b, c))[0]
k5im = lambda wre, wim, b, c: quad(func6im, 0, 1, args=(wre, wim, b, c))[0]
k6re = lambda wre, wim, b, c: quad(func7re, 0, 1, args=(wre, wim, b, c))[0]
k6im = lambda wre, wim, b, c: quad(func7im, 0, 1, args=(wre, wim, b, c))[0]
k7re = lambda wre, wim, b, c: quad(func8re, 0, 1, args=(wre, wim, b, c))[0]
k7im = lambda wre, wim, b, c: quad(func8im, 0, 1, args=(wre, wim, b, c))[0]
k8re = lambda wre, wim, b, c: quad(func9re, 0, 1, args=(wre, wim, b, c))[0]
k8im = lambda wre, wim, b, c: quad(func9im, 0, 1, args=(wre, wim, b, c))[0]
k9re = lambda wre, wim, b, c: quad(func10re, 0, 1, args=(wre, wim, b, c))[0]
k9im = lambda wre, wim, b, c: quad(func10im, 0, 1, args=(wre, wim, b, c))[0]
k10re = lambda wre, wim, b, c: quad(func11re, 0, 1, args=(wre, wim, b, c))[0]
k10im = lambda wre, wim, b, c: quad(func11im, 0, 1, args=(wre, wim, b, c))[0]
k11re = lambda wre, wim, b, c: quad(func12re, 0, 1, args=(wre, wim, b, c))[0]
k11im = lambda wre, wim, b, c: quad(func12im, 0, 1, args=(wre, wim, b, c))[0]
k12re = lambda wre, wim, b, c: quad(func13re, 0, 1, args=(wre, wim, b, c))[0]
k12im = lambda wre, wim, b, c: quad(func13im, 0, 1, args=(wre, wim, b, c))[0]
k13re = lambda wre, wim, b, c: quad(func14re, 0, 1, args=(wre, wim, b, c))[0]
k13im = lambda wre, wim, b, c: quad(func14im, 0, 1, args=(wre, wim, b, c))[0]
k14re = lambda wre, wim, b, c: quad(func15re, 0, 1, args=(wre, wim, b, c))[0]
k14im = lambda wre, wim, b, c: quad(func15im, 0, 1, args=(wre, wim, b, c))[0]
k15re = lambda wre, wim, b, c: quad(func16re, 0, 1, args=(wre, wim, b, c))[0]
k15im = lambda wre, wim, b, c: quad(func16im, 0, 1, args=(wre, wim, b, c))[0]
k16re = lambda wre, wim, b, c: quad(func17re, 0, 1, args=(wre, wim, b, c))[0]
k16im = lambda wre, wim, b, c: quad(func17im, 0, 1, args=(wre, wim, b, c))[0]
k17re = lambda wre, wim, b, c: quad(func18re, 0, 1, args=(wre, wim, b, c))[0]
k17im = lambda wre, wim, b, c: quad(func18im, 0, 1, args=(wre, wim, b, c))[0]
k18re = lambda wre, wim, b, c: quad(func19re, 0, 1, args=(wre, wim, b, c))[0]
k18im = lambda wre, wim, b, c: quad(func19im, 0, 1, args=(wre, wim, b, c))[0]
k19re = lambda wre, wim, b, c: quad(func20re, 0, 1, args=(wre, wim, b, c))[0]
k19im = lambda wre, wim, b, c: quad(func20im, 0, 1, args=(wre, wim, b, c))[0]
k20re = lambda wre, wim, b, c: quad(func21re, 0, 1, args=(wre, wim, b, c))[0]
k20im = lambda wre, wim, b, c: quad(func21im, 0, 1, args=(wre, wim, b, c))[0]
k21re = lambda wre, wim, b, c: quad(func22re, 0, 1, args=(wre, wim, b, c))[0]
k21im = lambda wre, wim, b, c: quad(func22im, 0, 1, args=(wre, wim, b, c))[0]
k22re = lambda wre, wim, b, c: quad(func23re, 0, 1, args=(wre, wim, b, c))[0]
k22im = lambda wre, wim, b, c: quad(func23im, 0, 1, args=(wre, wim, b, c))[0]
k23re = lambda wre, wim, b, c: quad(func24re, 0, 1, args=(wre, wim, b, c))[0]
k23im = lambda wre, wim, b, c: quad(func24im, 0, 1, args=(wre, wim, b, c))[0]
k24re = lambda wre, wim, b, c: quad(func25re, 0, 1, args=(wre, wim, b, c))[0]
k24im = lambda wre, wim, b, c: quad(func25im, 0, 1, args=(wre, wim, b, c))[0]
def det(w, b, c):
    G00 = 1+k1re(w[0], w[1], b, c)+1j*k1im(w[0], w[1], b, c)+k2re(w[0], w[1], b, c)+1j*k2im(w[0], w[1], b, c)+k3re(w[0], w[1], b, c)+1j*k3im(w[0], w[1], b, c)+k4re(w[0], w[1], b, c)+1j*k4im(w[0], w[1], b, c)
    G01 = k5re(w[0], w[1], b, c)+1j*k5im(w[0], w[1], b, c)+k6re(w[0], w[1], b, c)+1j*k6im(w[0], w[1], b, c)+k7re(w[0], w[1], b, c)+1j*k7im(w[0], w[1], b, c)+k8re(w[0], w[1], b, c)+1j*k8im(w[0], w[1], b, c)
    G02 = k9re(w[0], w[1], b, c)+1j*k9im(w[0], w[1], b, c)+k10re(w[0], w[1], b, c)+1j*k10im(w[0], w[1], b, c)+k11re(w[0], w[1], b, c)+1j*k11im(w[0], w[1], b, c)+k12re(w[0], w[1], b, c)+1j*k12im(w[0], w[1], b, c)
    G11 = -1+k13re(w[0], w[1], b, c)+1j*k13im(w[0], w[1], b, c)+k14re(w[0], w[1], b, c)+1j*k14im(w[0], w[1], b, c)+k15re(w[0], w[1], b, c)+1j*k15im(w[0], w[1], b, c)+k16re(w[0], w[1], b, c)+1j*k16im(w[0], w[1], b, c)
    G22 =  -1+k17re(w[0], w[1], b, c)+1j*k17im(w[0], w[1], b, c)+k18re(w[0], w[1], b, c)+1j*k18im(w[0], w[1], b, c)+k19re(w[0], w[1], b, c)+1j*k19im(w[0], w[1], b, c)+k20re(w[0], w[1], b, c)+1j*k20im(w[0], w[1], b, c)
    G12 = k21re(w[0], w[1], b, c)+1j*k21im(w[0], w[1], b, c)+k22re(w[0], w[1], b, c)+1j*k22im(w[0], w[1], b, c)+k23re(w[0], w[1], b, c)+1j*k23im(w[0], w[1], b, c)+k24re(w[0], w[1], b, c)+1j*k24im(w[0], w[1], b, c)
    V = np.array([[G00, G01, G02], [G01, G11, G12], [G02, G12, G22]])
    return (np.real(np.linalg.det(V)), np.imag(np.linalg.det(V)))
print(fsolve(det, (26, 6), args=(28*np.sin(math.pi*0.5*0.0+math.pi*0.5*0.0), 28*np.cos(math.pi*0.5*0.0+math.pi*0.5*0.0))))
#print(k5re(1, 0, 0, 0)+1j*k5im(1, 0, 0, 0)+k6re(1, 0, 0, 0)+1j*k6im(1, 0, 0, 0)+k7re(1, 0, 0, 0)+1j*k7im(1, 0, 0, 0)+k8re(1, 0, 0, 0)+1j*k8im(1, 0, 0, 0))
#print(k9re(1, 0, 0, 0)+1j*k9im(1, 0, 0, 0)+k10re(1, 0, 0, 0)+1j*k10im(1, 0, 0, 0)+k11re(1, 0, 0, 0)+1j*k11im(1, 0, 0, 0)+k12re(1, 0, 0, 0)+1j*k12im(1, 0, 0, 0))
#plt.plot(-k, arr2)
#plt.savefig("npy+1Ddispl9.png")
#np.save("kfull3left.npy", np.array(arr1))
#np.save("kfull4left.npy", np.array(arr2))
#print(arr1[39])
#print(arr2[39])"
arr1 = []
arr2 = []
theta = np.linspace(0, 1, 10)
for i in range(10):
    if(i==0):
        arr1.append(fsolve(det, [5.315088102586832e-14, 9], args=(13*np.sin(math.pi*0.5*theta[i]+math.pi*0.5*0), 13*np.cos(math.pi*0.5*theta[i]+math.pi*0.5*0)))[0])
        arr2.append(fsolve(det, [5.315088102586832e-14, 9], args=(13*np.sin(math.pi*0.5*theta[i]+math.pi*0.5*0), 13*np.cos(math.pi*0.5*theta[i]+math.pi*0.5*0 ))[1])
    else:
        arr1.append(fsolve(det, [arr1[i-1], arr2[i-1]], args=(13*np.sin(math.pi*0.5*theta[i]+math.pi*0.5*0), 13*np.cos(math.pi*0.5*theta[i]+math.pi*0.5*0))[0])
        arr2.append(fsolve(det, [arr1[i-1], arr2[i-1]], args=(13*np.sin(math.pi*0.5*theta[i]+math.pi*0.5*0), 13*np.cos(math.pi*0.5*theta[i]+math.pi*0.5*0))[1])
print(arr2)
print(arr1)            


