#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo methods to investigate the noise transfer and noise correlations in PET image reconstruction using MLEM 
# 

# ## Background

# Due to limitations in the amount of radiotracer that can be safely injected into a patient, the sensitivity of scanners and the available acquisition time, **acquired projection data in PET and SPECT** imaging usually suffer from **high levels of Poisson noise**. 
# 
# During the image reconstruction process, the **noise** of the data gets **transferred into the reconstructed images**. In clinical practice, the early-stopped iterative maximum likelihood expectation maximization (**MLEM**) algorithm is commonly used to reconstruct PET and SPECT images. When using early-stopped MLEM, an **analytic prediction** of the expected **noise level** in every voxel of the reconstructed image and the **noise correlations** between neighboring voxels is unfortunately **rather complicated**.

# ## Learning objetive

# The aim of this notebook is to learn how to investigate noise transfer and noise correlations using Monte Carlo methods. To do so, we will first simulate noise-free PET data. Second, we will generate many noise realizations by adding Poisson noise to the noise-free data. Last but not least, we will reconstruct all noise realizations using MLEM and study the noise properties by analyzing all reconstructions.

# **Additional background information (not needed to solve this notebook)** on MLEM is available in [this video](https://www.youtube.com/watch?v=CHKOSYdf47c) and in [this video](https://www.youtube.com/watch?v=Z70n5NCw9BY). Moreover, background information on the concept of maximum likelihood approaches are available [here](https://www.youtube.com/watch?v=uTa7g_h4c1E). Background information on PET and SPECT (**not needed to solve this notebook**)  is available [here](https://www.youtube.com/watch?v=M8DOzE2d0dw) and [here](https://www.youtube.com/watch?v=4mrtq8CeLvo&list=PLKkWkQgtnBS1tWAE3-TL1-MDKY9EUJTFP&index=2).

# **This notebook provides all python functions and classes needed to simulate and reconstruct "realistic" 2D PET data using MLEM. It contains two bigger programming tasks (including smaller sub-tasks) that have to be solved by you.**
# 
# **Task 1 focuses on the generation and analysis of Poisson noise realizations of PET data**.
# 
# **Task 2 focuses on the reconstruction of all these noise realization and the analysis of the noise transfer into
# the reconstructed images.**

# ### Module import section

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# import of modules that we need in this notebook
# make sure that utils.py is placed in the same directory 

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from utils import RotationBased2DProjector, PETAcquisitionModel, test_images, OS_MLEM

# needed to get inline matplotlib plots in an IPython notebook
plt.ion()

# ### Input parameter section

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# number of pixels for our images to be simulated and reconstructed
npix = 150
# pixel size in mm
pix_size_mm = 4

# number of iterations to use for OS-MLEM
num_iter    = 4
num_subsets = 20

# set the default color map to Greys (white to black) for showing images
plt.rcParams['image.cmap'] = 'Greys'


# ### Setup of ground truth images
# 
# We set up a simple elliptical 2D image representing a homogeneous uptake of a PET radiotracer in the abdomen and arms (```em_img```). Moreover, we also generate a corresponding attenuation image mimicking water attenuation inside the body (```att_img```). Note that the attenuation image is only needed in the forward model to model the effect of photon attenuation during data acquisition. During image reconstruction, we aim to recover the emission image from the acquired (noisy) projection data.

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# generate the ground truth activity (emission) and attenuation images that we use for the data simulation
em_img, att_img = test_images(npix)


# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# show the true activity (emission) image and the attenuation image
fig, ax = plt.subplots(1,2)
im0 = ax[0].imshow(em_img)
im1 = ax[1].imshow(att_img)
fig.colorbar(im0, ax = ax[0], location = 'bottom')
fig.colorbar(im1, ax = ax[1], location = 'bottom')
ax[0].set_title('ground truth activity image')
ax[1].set_title('ground truth attenuation image')
fig.tight_layout()

print(f'image shape {em_img.shape}')


# ### Simulation of noise-free data

# Based on an acquisition model that includes the physics of the data acquisition process of a simplified 2D PET system, we can now simulate noise free data.

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# setup the forward projector"
# the acq_model object is an abstract representation of the linear operator P (and also it's adjoint)
proj = RotationBased2DProjector(npix, pix_size_mm = pix_size_mm, num_subsets = num_subsets)
acq_model = PETAcquisitionModel(proj, att_img)


# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# generate noise free data by applying the acquisition model to our simulated emission image
noise_free_data = acq_model.forward(em_img)


# The simulated PET data is a 2D array called a *sinogram*. Every row in this sinogram contains a (corrected) parallel forward projection of our ground truth object. The dimension of the sinogram array should (180 views, 150 radial elements).

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# show the noise-free and noisy simulated emission data (sinogram)
fig2, ax2 = plt.subplots(1,1, figsize = (3,6))
im2 = ax2.imshow(noise_free_data, vmin = 0, vmax = 65)
fig2.colorbar(im2, ax = ax2, location = 'bottom')
ax2.set_xlabel('radial element')
ax2.set_ylabel('view')
ax2.set_title('noise-free data')
fig2.tight_layout()

print(f'data (sinogram) shape {noise_free_data.shape}')


# ### Adding noise to the noise-free data

# Based on the physics of the acquisition process (photon counting) we know that the acquired data in PET can be very well described by independent Poisson distributions. (More detailed information on why that is true can be e.g. found in this [short video](https://www.youtube.com/watch?v=QD8iekOc0u8)). To sample from independent Poisson distributions with known mean (the known mean value in every data bin of the 2D sinogram is the value obtained from the simulated noise-free data), we can use the function ```np.random.poisson``` from numpy's random module. To obtain reproducible results, we have set the **seed of the random generator** using ```np.random.seed()```. Note that if we would not set the seed explicitly, every time we would re-run the cell below, we would get a different noise realization of the same noise-free data. 

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# add poisson noise to the data
np.random.seed(1)
noisy_data_1 = np.random.poisson(noise_free_data)


# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# show the noise-free and noisy simulated emission data (sinogram)
fig3, ax3 = plt.subplots(1,2, figsize = (6,6))
im02 = ax3[0].imshow(noise_free_data, vmin = 0, vmax = 65)
im12 = ax3[1].imshow(noisy_data_1, vmin = 0, vmax = 65)
fig3.colorbar(im02, ax = ax3[0], location = 'bottom')
fig3.colorbar(im12, ax = ax3[1], location = 'bottom')
ax3[0].set_xlabel('radial element')
ax3[1].set_xlabel('radial element')
ax3[0].set_ylabel('view')
ax3[1].set_ylabel('view')
ax3[0].set_title('noise-free data')
ax3[1].set_title('noisy data - 1st noise realization')
fig3.tight_layout()

print(f'data (sinogram) shape {noise_free_data.shape}')


# Let's generate a second noise realization using a different seed for numpy's random generator.

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# add poisson noise to the data
np.random.seed(2)
noisy_data_2 = np.random.poisson(noise_free_data)


# Let's display the noise-free data, the first two noise realizations and the difference between both noise realizations to convince ourselves that the noise realizations are indeed different.

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# show the noise-free and noisy simulated emission data (sinogram)
fig4, ax4 = plt.subplots(1,4, figsize = (12,6))
im04 = ax4[0].imshow(noise_free_data, vmin = 0, vmax = 75)
im14 = ax4[1].imshow(noisy_data_1, vmin = 0, vmax = 75)
im24 = ax4[2].imshow(noisy_data_2, vmin = 0, vmax = 75)
im34 = ax4[3].imshow(noisy_data_2 - noisy_data_1, vmin = -30, vmax = 30, cmap = plt.cm.seismic)
fig4.colorbar(im04, ax = ax4[0], location = 'bottom')
fig4.colorbar(im14, ax = ax4[1], location = 'bottom')
fig4.colorbar(im24, ax = ax4[2], location = 'bottom')
fig4.colorbar(im34, ax = ax4[3], location = 'bottom')

for axx in ax4:
  axx.set_xlabel('radial element')
  axx.set_ylabel('view')

ax4[0].set_title('noise-free data')
ax4[1].set_title('noisy data - 1st noise realization')
ax4[2].set_title('noisy data - 2nd noise realization')
ax4[3].set_title('2nd - 1st noise realization')
fig4.tight_layout()

print(f'data (sinogram) shape {noise_free_data.shape}')


# ## Task 1 - Your Turn
# 
# - 1.1: Generate $n$ = 200 Poisson noise realizations of the simulated noise-free data
# - 1.2: Plot the value of the data bin [90,75] (the central bin of the 2D sinogram) for all $n$ noise realizations
# - 1.3: calculate the mean value and standard deviation of data bin [90,75] over all $n$ noise realizations. What values for the mean and standard variation of data bin [90,75] do you expect?
# - 1.4: Plot the all values of the data bin [90,75] against the all values of the data bin [90,76] and calculate the Pearson correlation coefficient between them. What do you expect for the correlation coefficient and why? To calculate the Pearson correlation coefficient, use the first argument returned ```pearsonr()``` - see [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html).

# In[ ]:


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
# ADD THE SOURCE CODE FOR TASK 1.1, 1.2, 1.3, and 1.4 HERE

#num_real = 200 # number of noise realizations

## allocate an array of shape [num_real, 180, 100] for all noise realizations
#all_noise_realizations = np.zeros((num_real,) + noise_free_data.shape, dtype = np.uint32)

## generate num_real noise realizations in a loop
#for i in range(num_real):
#    all_noise_realizations[i,...] = ...

# ...
# ...
# ...


#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------


# In[ ]:


# UNCOMMENT THE FOLLOWING LINE AND EXECUTE TO CELL TO SEE THE SOLUTION
#%load ../../teaching_notebooks_solutions/MC_MLEM_noise_transfer/sol_1.py


# ## Image reconstruction using MLEM
# 
# 

# After having simulated different noise realizations of "realistic" 2D PET data, we can now use MLEM (with ordered subsets) to reconstruct each noise realization. To reconstruct a single noise realization (e.g. ```noisy_data_1```), you can an instance of the ```OS_MLEM``` class as shown in the next cell. In this notebook we use MLEM with 4 iterations with 20 subsets which means that image 80 updates are performed. Depending on your computing hardware, this should take ca. 1-3s.

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# reconstruct the first noise realization
reconstructor_1 = OS_MLEM(noisy_data_1, acq_model)
recon_1 = reconstructor_1.run(num_iter, verbose = True)


# Let's also reconstruct a different noise realization of the same noise-free data (```noisy_data_2```).

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# reconstruct the second noise realization
reconstructor_2 = OS_MLEM(noisy_data_2, acq_model)
recon_2 = reconstructor_2.run(num_iter, verbose = True)


# Let's display the grond truth emission image (the image used to simulate the noise-free data) and the reconstructions of the two noise realizations and their difference.

# In[ ]:


# RUN, BUT DO NOT CHANGE THIS CELL

# show the noise-free and noisy simulated emission data (sinogram)
fig7, ax7 = plt.subplots(1,4, figsize = (12,6))
im07 = ax7[0].imshow(em_img,  vmin = 0, vmax = 1.4*em_img.max())
im17 = ax7[1].imshow(recon_1, vmin = 0, vmax = 1.4*em_img.max())
im27 = ax7[2].imshow(recon_2, vmin = 0, vmax = 1.4*em_img.max())
im37 = ax7[3].imshow(recon_2 - recon_1, vmin = -0.5*em_img.max(), vmax = 0.5*em_img.max(), cmap = plt.cm.seismic)
fig7.colorbar(im07, ax = ax7[0], location = 'bottom')
fig7.colorbar(im17, ax = ax7[1], location = 'bottom')
fig7.colorbar(im27, ax = ax7[2], location = 'bottom')
fig7.colorbar(im37, ax = ax7[3], location = 'bottom')
ax7[0].set_title('ground truth image')
ax7[1].set_title('reconstr. 1st noise real')
ax7[2].set_title('reconstr. 2nd noise real')
ax7[3].set_title('2nd recon - 1st recon')
fig7.tight_layout()


# ## Task 2 - Your turn
# 
# - 2.1: Reconstruct all $n = 200$ noise realizations and store the all 2D reconstructions in a single 3D array.
# - 2.2: Calculate a 2D image (array) representing the mean reconstructed value across all $n = 200$ reconstructions in every pixel. To do so, look at the ``mean()`` method of numpy's ``ndarray`` class [here](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html).
# - 2.3: Calculate a 2D image (array) representing the standard deviations of the reconstructed values across all $n = 200$ reconstructions in every pixel. To do so look at the ``std()`` method of numpy's ``ndarray`` class [here](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.std.html). Use ```ddof = 1```, to get the unbiased estimate for the sample standard variation. **This image is a Monte Carlo estimate of the expected noise level in every pixel of the reconstruction.**
# - 2.4: Display the ground truth image, the reconstruction of the first noise realization, the "mean" image and the "standard deviation/noise level" image next to each other. What do you observe in the "standard deviation/noise level" image. Is the estimated noise level homogenous across the whole image?
# - 2.5: Calculate the Pearson correlation coefficient between the 200 reconstructed values in the central pixels [75,75] and the 200 reconstructed values of its neighboring pixels [75,75+k] for k = -5,-4,-3,-2,-1,1,0,1,2,3,4,5. Plot the correlation coefficients as a function of k. What do you observe for the noise correlation between the central pixel [75,75] and its neighboring pixels?
# - 2.6: Visualize the noise correlations with 4 scatter plots where you plot the 200 values of pixel [75,75] against the 200 values of pixel [75,75+j] for j = 1,2,3,4.

# In[ ]:


#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------- 
# ADD YOUR CODE FOR TASK 2.1 HERE

## allocate array to store all reconstructions
#recons = np.zeros((num_real,) + (npix, npix))

#for i in range(num_real):
#    print(f'reconstruction of noise realization {(i+1):04}', end = '\r')    
#    ...
#    ...
#    recons[i,...] = ...
    
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------


# In[ ]:


# UNCOMMENT THE FOLLOWING LINE AND EXECUTE TO CELL TO SEE THE SOLUTION
#%load ../../teaching_notebooks_solutions/MC_MLEM_noise_transfer/sol_2_1.py


# In[ ]:


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# ADD YOUR CODE FOR TASK 2.2, 2.3 and 2.4 HERE



#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# In[ ]:


# UNCOMMENT THE FOLLOWING LINE AND EXECUTE TO CELL TO SEE THE SOLUTION
#%load ../../teaching_notebooks_solutions/MC_MLEM_noise_transfer/sol_2_2.py


# In[ ]:


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# ADD YOUR CODE FOR TASK 2.5 HERE



#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------


# In[ ]:


# UNCOMMENT THE FOLLOWING LINE AND EXECUTE TO CELL TO SEE THE SOLUTION
#%load ../../teaching_notebooks_solutions/MC_MLEM_noise_transfer/sol_2_5.py


# In[ ]:


#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
# ADD YOUR CODE FOR TASK 2.6 HERE



#---------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------


# In[ ]:


# UNCOMMENT THE FOLLOWING LINE AND EXECUTE TO CELL TO SEE THE SOLUTION
#%load ../../teaching_notebooks_solutions/MC_MLEM_noise_transfer/sol_2_6.py

