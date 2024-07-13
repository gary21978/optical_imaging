from matplotlib import pyplot as plt
from pupil import Pupil
from lightsource import LightSource
from mask import Mask
import numpy as np
import torch
from imageformation import abbeImage
import meent

def createGeometry():
    from PIL import Image, ImageDraw, ImageFont
    image = np.zeros((64, 64), dtype=np.uint8)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 48)
    text = "T"
    draw.text((15, 5), text, fill=1, font=font)
    return np.array(image, dtype=np.int8)

def nearfieldImaging(geometry, illuminant, computation, device=None):
    if device is None:
        device = torch.device('cpu')

    # reflective indices
    n_OX = 1.4745
    n_TIN = 2.0669 + 1.0563j
    n_TEOS = 1.380
    n_SIN = 2.1222
    n_substrate = 6.5271 + 2.6672j

    if illuminant['polarization'] == 'TE':
        pol = 0
    else:
        if illuminant['polarization'] == 'TM':
            pol = 1
        else:
            raise ValueError('Polarization must be TE or TM')

    n_I = 1
    n_II = n_substrate
    theta = illuminant['polar_angle']
    phi = illuminant['azimuthal_angle']

    M = geometry
    #thickness = [5, 21, 100, 20, 5]
    #ns = [n_OX, n_TIN, n_TEOS, n_SIN, n_OX]
    #Ms = [M, M, M, np.ones_like(M), np.ones_like(M)]
    thickness = [5, 21, 100] 
    ns = [n_OX, n_TIN, n_TEOS]
    Ms = [M, M, M]

    period = [geometry.shape[0]*computation['pixel_size'], geometry.shape[1]*computation['pixel_size']]
    fourier_order = [computation['truncation_order'], computation['truncation_order']]
    ucell = np.array([M*(n**2-1) + 1 for n, M in zip(ns, Ms)])

    # backend numpy 0 PyTorch 2
    mee = meent.call_mee(backend=2, device=device, grating_type=2, pol=pol, n_I=n_I, n_II=n_II, \
                        theta=theta, phi=phi, fourier_order=fourier_order, wavelength=illuminant['wavelength'], \
                        period=period, ucell=ucell, thickness=thickness, type_complex=np.complex128, \
                        fft_type=0, improve_dft=True)

    _, _, field_cell = mee.conv_solve_field(res_z=2, \
                                            res_y=computation['pixel_number'], \
                                            res_x=computation['pixel_number'])
    Efield = np.squeeze(field_cell[0, :, :, 0:3])
    return Efield

def farfieldImaging(Efield, illuminant, pupil, device=None):
    if device is None:
        device = torch.device('cpu')

    lightsource = LightSource(sigmaIn=illuminant['sigma_in'], sigmaOut=illuminant['sigma_out'], \
                              pixelNumber=computation['pixel_number'], device=device)
    ls = lightsource.generateAnnular()
    pupil = Pupil(pixelNumber=computation['pixel_number'], wavelength=illuminant['wavelength'], NA = pupil['NA'], \
                  aberrations=pupil['aberrations'], obstruction=0.3, magnification=1.0, device=device)
    pupilFunction = pupil.generatePupilFunction()
    intensity = 0
    for xyz in range(3):
        mask = Mask(geometry=Efield[: , :, xyz], device=device)
        maskFT = mask.fraunhofer(illuminant['wavelength'], True)
        aerialImage = abbeImage(mask, maskFT, pupilFunction, ls, mask.pixelSize, mask.deltaK, \
                                illuminant['wavelength'], device)
        intensity = intensity + aerialImage
    return intensity.cpu(), pupilFunction.cpu()

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    geometry = createGeometry()
    illuminant = {'polarization': 'TE', \
                  'polar_angle': 0, \
                  'azimuthal_angle': 0, \
                  'wavelength': 365, \
                  'sigma_in': 0.1, \
                  'sigma_out': 0.3}
    # The coherence factor is defined as the spot diameter of the 
    # lightsource (or sin(theta) of the max incident angle)
    # divided by the diameter of the entrance pupil (or projection NA).
    # Lower simga means better contrast
    pupil = {'obstruction': 0.3, \
             'aberrations': [0, 0, 0, 0, 1, 0.03, 0.02, 0.01, 0.01, 0.01], \
             'NA': 0.7, \
             'magnification': 1.0}
    computation = {'truncation_order': 3, 'pixel_number': 64, 'pixel_size': 20}

    Efield = nearfieldImaging(geometry, illuminant, computation, device=device)
    intensity, PF = farfieldImaging(Efield, illuminant, pupil, device=device)

    Lx = geometry.shape[0]*computation['pixel_size']
    Ly = geometry.shape[1]*computation['pixel_size']

    LI = computation['pixel_number']*computation['pixel_size'] #????
    
    fig, axes = plt.subplots(2, 4)
    axes[0][0].imshow(geometry, cmap='gray', interpolation='none', extent = (0, Lx, 0, Ly))
    axes[0][0].set_title('design')
    axes[0][0].set_xlabel('nm')
    axes[0][0].set_ylabel('nm')

    axes[0][1].imshow(intensity, cmap='gray', interpolation='none', extent = (0, LI, 0, LI))
    axes[0][1].set_title('intensity')
    axes[0][1].set_xlabel('nm')
    axes[0][1].set_ylabel('nm')

    axes[0][2].imshow(np.real(PF), cmap='gray')
    axes[0][2].set_title('Pupil function (real part)')

    axes[0][3].imshow(np.imag(PF), cmap='gray')
    axes[0][3].set_title('Pupil function (imaginary part)')

    titles = ['|Ex|', '|Ey|', '|Ez|']
    for k in range(3):
        P = axes[1][k].imshow(np.abs(Efield[:,:,k]), cmap='jet', \
                              extent = (0, Lx, 0, Ly))
        axes[1][k].set_title(titles[k])
        #fig.colorbar(P, ax=axes[1][k])
        axes[1][k].set_xlabel('nm')
        axes[1][k].set_ylabel('nm')

    I = np.abs(Efield[:,:,0])**2 + np.abs(Efield[:,:,1])**2 +np.abs(Efield[:,:,2])**2
    axes[1][3].imshow(I, cmap='gray', interpolation='none', extent = (0, Lx, 0, Ly))
    axes[1][3].set_title('|E|^2')
    axes[1][3].set_xlabel('nm')
    axes[1][3].set_ylabel('nm')
    plt.show()