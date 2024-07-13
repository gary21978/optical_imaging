import torch

class LightSource:

    def __init__(self, sigmaIn: torch.float16=0, sigmaOut: torch.float16=0.6, pixelNumber: int=64, \
                 NA: torch.float16=0.7, shiftX: int = 0, shiftY: int = 0, device: torch.device=None):

        if type(device) is torch.device:
            self.device = device
        elif torch.cuda.is_available:
            self.device = torch.device('cuda')
            print(f"No device defined for light source! Using {torch.cuda.get_device_name(self.device)}.")
        else:
            self.device = torch.device('cpu')
            print(f"No device defined for light source! Using CPU.")

        self.pixelNumber = pixelNumber
        self.NA = NA #being the projection NA
        
        #When we use sigma as a radius like this, we are referring to the partial coherence factor. This tends to be defined as the spot diameter of the lightsource (or sin(theta) of the max incident angle)
        #divided by the projection NA or diameter of the entrance pupil. Since, in the Abbe formulation of PCI, each point generates an image, lower sigma means less "smear" (read: better contrast).
        #in a physics sense, our diffraction patterns can be seen as smeared shadows of our extended light source, with larger sources leaving less contrast because of the larger angles of incidence
        #https://www.lithoguru.com/scientist/CHE323/Lecture45.pdf, chapter 12 of [7] and chapter 2 of [8] also have some good insights.
        #For annular/quasar sources the inner and outer are defined both in the terms of pcf, that is, in terms of the ratio the inner radius and outer radius take of the NAc/NAp
        self.sigmaInner = sigmaIn 
        self.sigmaOuter = sigmaOut

        self.shiftX = shiftX #Units of wavelength/NA, so if we have 1 for this shift value it will shift by an entire sigma -- that is, ending up on the edge of the pupil.
        self.shiftY = shiftY

    def generateAnnular(self) -> torch.Tensor:

        sigmaSpan = 2 #We want to show from +2sigma to -2sigma, such that -1 to 1 are shown in the center for the pupil function
        deltaSigma = sigmaSpan*2/self.pixelNumber #step size, since we want a tensor of pixelNumber size, and sigmaSpan is half

        sigmaX = torch.arange(-sigmaSpan-self.shiftX, sigmaSpan-self.shiftX, deltaSigma, dtype=torch.float16, device=self.device)
        sigmaY = torch.arange(-sigmaSpan-self.shiftY, sigmaSpan-self.shiftY, deltaSigma, dtype=torch.float16, device=self.device)
        #Units of wavelength/NA, where the unit circle sigma=1 will always be the lens apeture size. Refer to eq 2.86 from mack
        #The key here is that when we calculate the Zernike polynomial, it exists on the unit circle, while our light source (provided sigma<1) will exist within the pupil space
        #We don't need fancy unit juggling here or within the pupil function, because we define the units now and can convert to xy using the NA and wavelength later

        sX, sY = torch.meshgrid((sigmaX, sigmaY), indexing='xy')
        O = torch.sqrt(sX**2 + sY**2)

        area = 3.1416*(self.sigmaOuter**2 - self.sigmaInner**2)
        lightsource = torch.where((O >= self.sigmaInner) & (O <= self.sigmaOuter), 1/area, 0)

        return lightsource
    
    def generateQuasar(self, count, rotation) -> torch.Tensor:

        sigmaSpan = 2
        deltaSigma = sigmaSpan*2/self.pixelNumber

        sigmaX = torch.arange(-sigmaSpan-self.shiftX, sigmaSpan-self.shiftX, deltaSigma, dtype=torch.float16, device=self.device)
        sigmaY = torch.arange(-sigmaSpan-self.shiftY, sigmaSpan-self.shiftY, deltaSigma, dtype=torch.float16, device=self.device)

        sX, sY = torch.meshgrid((sigmaX, sigmaY), indexing='xy')
        O = torch.sqrt(sX**2 + sY**2)
        theta = torch.atan2(sY, sX) + rotation #angle to any given spot from origin
        theta %= 2*torch.pi #wrap around to stay on the unit circle

        annularForm = torch.where((O >= self.sigmaInner) & (O <= self.sigmaOuter), 1, 0)

        angularSpacing = (torch.pi / count)
        lightsource = annularForm

        for gap in range(count):
            lightsource = lightsource * torch.where(((gap+gap)*angularSpacing<theta) & (theta<(gap+gap+1)*angularSpacing), 0, 1)

        return lightsource