import torch
from math import factorial, sqrt, ceil

class Pupil:

    def __init__(self, pixelNumber: int = 64, wavelength: int = 193, NA: torch.float16 = 0.7, aberrations: torch.Tensor = None, \
                 obstruction: torch.float16 = 0, magnification: torch.float16 = 1.0, device: torch.device=None):

        if type(device) is torch.device:
            self.device = device
        elif torch.cuda.is_available:
            self.device = torch.device('cuda')
            print(f"No device defined for pupil function! Using {torch.cuda.get_device_name(self.device)}.")
        else:
            self.device = torch.device('cpu')
            print("No device defined for pupil function! Using CPU.")

        if aberrations is None:
            print("No aberrations defined for pupil function! Assuming perfect system.")
            self.aberrations = [0] #assume perfect lens, so piston=0 and nothing more
        else:
            self.aberrations = aberrations

        self.pixelNumber = pixelNumber

        self.wavelength = wavelength
        self.NA = NA #being the projection NA
        self.obstruction = obstruction
        self.magnification = magnification
        
    def generatePupilFunction(self) -> torch.Tensor:
        WE = generateWavefrontError(self.aberrations, self.pixelNumber, self.NA, self.wavelength, self.obstruction, self.device)
        phi = generatePhi(WE, self.pixelNumber, self.obstruction, self.device)
        RC = RadiometricCorrection(self.magnification, self.pixelNumber, self.device)
        return phi*RC
    
    def generateWavefrontError(self) -> torch.Tensor:
        return generateWavefrontError(self.aberrations, self.pixelNumber, self.NA, self.wavelength, self.obstruction, self.device)
    
def diracd(v):
    if v == 0:
        return 1
    else:
        return 0

def generateZ(m, n, pixelNumber, coeff, obstruction=0, device=None):
    #implementation of eq (4.37) from [2], normalization factor from [5]
    #kudos to [5] for some hints on this

    sigmaSpan = 2
    deltaSigma = sigmaSpan*2/pixelNumber

    x = torch.arange(-sigmaSpan, sigmaSpan, deltaSigma, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid((x, x), indexing='xy')

    r = torch.sqrt(X**2 + Y**2)
    theta = torch.arctan2(Y, X)

    lLim = int((n-abs(m))/2)
    ilLim = int((n+abs(m))/2)

    Rmn = torch.zeros((lLim+1, pixelNumber, pixelNumber), dtype=torch.float16, device=device)
    for k in range(lLim+1):
        staticCoeff = ((-1)**k * factorial(n-k)) / (factorial(k)*factorial(ilLim-k)*factorial(lLim-k))
        intm = staticCoeff * r**(n-2*k)
        Rmn[k] = intm

    R = torch.sum(Rmn, dim=0)
    Nmn = sqrt((2*n+1)/(1+diracd(m)))

    if m >= 0:
        Z = coeff * Nmn * R * torch.cos(m*theta)
    else:
        Z = coeff * -Nmn * R * torch.sin(m*theta)

    Z = torch.where(r<=1, Z, 0)
    Z = torch.where(r>=obstruction, Z, 0)
    return Z

def OSA(m, n):
    return (n*(n+2)+m)/2

def OSAindexToMN(ji): #TODO: add the annoying fringe indexing system
    #eq (4.39) and (4.40) in [2]
    n = ceil(1/2*(-3 + sqrt(9 + 8*ji))) 
    m = (2*ji) - (n*(n + 2))
    return m, n

def generateWavefrontError(aberrations, pixelNumber, NA, wavelength, obstruction, device):
    WE = torch.zeros((pixelNumber, pixelNumber), dtype=torch.float16, device=device)

    if(len(aberrations)>=4):
        aberrations[4] = aberrations[4]*NA**2/(4*wavelength) #eq (3.24) of [8]

    for i in range(len(aberrations)):
        m, n = OSAindexToMN(i)
        coeff = aberrations[i]
        Z = generateZ(m, n, pixelNumber, coeff, obstruction, device)
        WE = WE + Z

    return WE.type(torch.complex64)

def generatePhi(WE, pixelNumber, obstruction, device):
    phi = torch.exp(1j*2*torch.pi*WE)

    sigmaSpan = 2
    deltaSigma = sigmaSpan*2/pixelNumber
    x = torch.arange(-sigmaSpan, sigmaSpan, deltaSigma, dtype=torch.float16, device=device)
    X, Y = torch.meshgrid((x, x), indexing='xy')
    r = torch.sqrt(X**2 + Y**2)

    phi = torch.where(r<=1, phi, 0)
    phi = torch.where(r>=obstruction, phi, 0)
    return phi

def RadiometricCorrection(magnification, pixelNumber, device):
    sigmaSpan = 2
    deltaSigma = sigmaSpan*2/pixelNumber
    x = torch.arange(-sigmaSpan, sigmaSpan, deltaSigma, dtype=torch.float16, device=device)
    X, Y = torch.meshgrid((x, x), indexing='xy')
    r = torch.sqrt(X**2 + Y**2)

    eps = 1e-6
    A = torch.where(1 - r**2*magnification > 0, 1 - r**2*magnification, eps)
    B = torch.where(1 - r**2 > 0, 1 - r**2, eps)
    rc = torch.pow(A/B, 0.25)
    return rc