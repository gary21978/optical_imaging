import torch
from mask import Mask

def calculateFFTAerial(pf, maskFFFT, pixelNumber, epsilon, N, device):
    pfAmplitudeProduct = pf * maskFFFT
    paddingWidth = (N - pixelNumber) // 2
    padder = torch.nn.ConstantPad2d(paddingWidth, 0)
    paddedPFA = padder(pfAmplitudeProduct)

    standardFormPPFA = torch.fft.fftshift(paddedPFA) #back into fft order
    abbeFFT = torch.fft.ifft2(standardFormPPFA, s=(N, N), norm='forward') #TODO: why is this ifft2 instead of fft2 like it is in the matlab source code? Bizzare offset otherwise
    unrolledFFT = torch.fft.ifftshift(abbeFFT)
    usqAbbe = torch.abs(unrolledFFT.unsqueeze(0).unsqueeze(0)).to(torch.float32)
    aerial = torch.nn.functional.interpolate(usqAbbe, scale_factor=(1/epsilon), mode='bilinear').to(torch.float32).squeeze(0).squeeze(0)  
    aerial = aerial**2

    extraSize = int((aerial.size()[0] - (pixelNumber+(2*paddingWidth))) / 2 + paddingWidth) #TODO: Make this not bad
    trimmedAerial = aerial[extraSize:extraSize+pixelNumber, extraSize:extraSize+pixelNumber]

    return trimmedAerial

def abbeImage(mask, maskFT: torch.Tensor, pupilF: torch.Tensor, lightsource: torch.Tensor, pixelSize: int, deltaK: float, wavelength:int, device: torch.device):

    epsilon, N = Mask.calculateEpsilonN(self=mask, deltaK=deltaK, pixelSize=pixelSize, wavelength=wavelength)
    pixelNumber = maskFT.size()[0]
    #fraunhoferConstant = (-2*1j*torch.pi)/wavelength

    image = torch.zeros((pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    pupilOnDevice = pupilF.to(device)
    pupilshift = torch.zeros((pixelNumber*2, pixelNumber*2, pixelNumber, pixelNumber), dtype=torch.complex64, device=device)

    a = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    b = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    A, B = torch.meshgrid((a, b), indexing='xy')

    i = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    j = torch.arange(0, pixelNumber, 1, dtype=int, device=device)
    I, J = torch.meshgrid((i, j), indexing='xy')
    
    Iu = I.unsqueeze(-1).unsqueeze(-1)
    Ju = J.unsqueeze(-1).unsqueeze(-1)
    
    pupilshift[A+Iu, B+Ju, Iu, Ju] = pupilOnDevice
    #there are Px x Px fields of Px x Px (1) where each (1) field has the pupil function where it is illuminated by the light source at a different position within it.
    # A and B represent every position in our un-padded field, and I and J respectively slide the pupil around our padded pupilshift space by broadcasting the AB grid across itself grid through addition
    # such that A begins at 1 for I = 1, etc.
    psTrim = pupilshift.narrow(0, pixelNumber//2 - 1, pixelNumber).narrow(1, pixelNumber//2 - 1, pixelNumber)

    for i in range(pixelNumber):
        for j in range(pixelNumber):
            if (lightsource[i, j] > 0):
                new = calculateFFTAerial(psTrim[:, :, j, i], maskFT, pixelNumber, epsilon, N, device)
                image = image + new

    return torch.abs(image)