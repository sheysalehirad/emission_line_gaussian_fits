
###########################################################################################
"""

Fitting_Gaussian_models_to_doublet_emission_lines.py      Created By: Sheyda Salehirad February 2020

This program fits one-component Gaussian models to emission line doublets. The example below is for the [S II]6716,6731 doublet lines, but they can easily be replaced for any other emission lines.

"""

import numpy as np
import pandas as pd
from lmfit.models import LinearModel, GaussianModel
from lmfit import Model, Parameters , Parameter, fit_report

def sii_simple(spectra , lamda):
    """
    For arrays of flux density and wavelength, this code returns one-component Gaussian fits to any double emission lines.
    The output consists of a chunk of the spectrum around the lines of interest (flux density and wavelength), component fit arrays, and a dictionary of the parameters that the fit produces.
    
    parameters:
    specra: array
        flux density 
    lamda: array
        wavelength
        
    """
    
    
    continuum = 40 # number of pixels to be selected around the center of the emission lines
    
    #selecting the spectrum around the emission lines
    SII = (lamda > 6716 - continuum) & (lamda < 6731 + continuum)
    x = lamda[SII]  
    y = spectra[SII]
    
    #adding a linear model to the fit to account for any continuum residuals
    linear = LinearModel(prefix = 'linsii')
    pars = linear.guess(y, x=x)
    
    gauss1 = GaussianModel(prefix = 'gs1_')
    pars.update(gauss1.make_params())

    pars['gs1_center'].set(value = 6716.31 , min = 6711, max = 6721,  vary = True)
    pars['gs1_sigma'].set(value = 2.2, min = 1.7 ,max = 5, vary = True)
    pars['gs1_amplitude'].set(value= pars['gs1_sigma'] * np.abs(np.max(y)) * 2.5, min = 0 , vary = True) 

    gauss2 = GaussianModel(prefix='gs2_')
    pars.update(gauss2.make_params())

    pars['gs2_center'].set(expr='gs1_center + 14.38') # the distance between the center of the lines in rest frame, values were taken from "Table of Spectral Lines Used in SDSS".
    pars['gs2_sigma'].set(expr='gs2_center * (gs1_sigma / gs1_center)') # the width are the same in the velocity space
    pars['gs2_amplitude'].set(value= pars['gs1_amplitude'] , min = 0, vary = True) $ no constraint on the heights of the lines
    
    model = gauss1 + gauss2 + linear
    
    result= model.fit(y , pars , x=x)
    comps = result.eval_components(x=x) # each component of the model

    #the parameters returned by the GaussianModel and LinearModel in lmfit
    std= {}
    for key in result.params:
        std.update({'std_{}'.format(key):result.params[key].stderr,'{}'.format(key):result.params[key].value})
    
    # a dictionary of some of the parameters of interest outputted by lmfit
    res= {**std, 'chi2sii': result.chisqr, 'redchi2sii': result.redchi, 'dofsii':result.nfree, 'aicsii': result.aic, 'bicsii':result.bic, 'successsii':result.success}  
    return y, x , comps, res

