# TODO: - check adjointness (not perfect yet)
#       - subset projection and OSEM

import numpy as np
from scipy.ndimage import rotate, gaussian_filter
from abc import ABC, abstractmethod


def test_images(num_pix: int,
                bg_activity: float = 2.,
                insert_activity: float = 2.,
                D0: float = 0.4,
                D1: float = 0.7,
                D_insert: float = 0.05,
                D_arm: float = 0.12,
                offset_arm: float = 0.43) -> tuple[np.ndarray, np.ndarray]:
    """
  create emission and attenuation image for 2D "human ellipsoid phantom"
  """

    img_shape = (num_pix, num_pix)
    em_img = np.zeros(img_shape)
    att_img = np.zeros(img_shape)

    x = np.linspace(-0.5, 0.5, num_pix)
    X0, X1 = np.meshgrid(x, x, indexing='ij')

    R_ell = np.sqrt((2 * X0 / D0)**2 + (2 * X1 / D1)**2)
    em_img[R_ell <= 1] = bg_activity

    att_img[R_ell <= 1] = 0.01

    # add insert
    R = np.sqrt(X0**2 + X1**2)
    em_img[R <= 0.5 * D_insert] = insert_activity

    # add arms
    R_left_arm = np.sqrt(X0**2 + (X1 - offset_arm)**2)
    em_img[R_left_arm <= 0.5 * D_arm] = bg_activity
    att_img[R_left_arm <= 0.5 * D_arm] = 0.01

    R_right_arm = np.sqrt(X0**2 + (X1 + offset_arm)**2)
    em_img[R_right_arm <= 0.5 * D_arm] = bg_activity
    att_img[R_right_arm <= 0.5 * D_arm] = 0.01

    return em_img, att_img


#---------------------------------------------------------------------------------------------------


class LinearSubsetOperator(ABC):
    @abstractmethod
    def forward(self, img: np.ndarray) -> np.ndarray:
        """ complete forward step """

    @abstractmethod
    def forward_subset(self, img: np.ndarray, subset: int) -> np.ndarray:
        """ subset forward step """

    @abstractmethod
    def adjoint(self, sino: np.ndarray) -> np.ndarray:
        """ adjoint of forward step """

    @abstractmethod
    def adjoint_subset(self, sino: np.ndarray, subset: int) -> np.ndarray:
        """ adjoint of susbet forward step """


#---------------------------------------------------------------------------------------------------


class RotationBased2DProjector(LinearSubsetOperator):
    """ 2D Rotation based projector

    Parameters
    ----------

    num_pix ... uint 
                number of pixels (the images to be projected have shape (num_pix, num_pix)

    pix_size_mm ... float
                    the pixel size in mm

    """
    def __init__(self,
                 num_pix: int,
                 num_views: int = 180,
                 pix_size_mm: float = 3.,
                 num_subsets: int = 20) -> None:

        self.num_pix = num_pix
        self.pix_size_mm = pix_size_mm
        self.img_shape = (self.num_pix, self.num_pix)
        self.num_views = num_views
        self.projection_angles = np.linspace(0,
                                             180,
                                             self.num_views,
                                             endpoint=False)
        self.sino_shape = (self.num_views, self.num_pix)
        self.num_subsets = num_subsets

        self.rotation_kwargs = dict(reshape=False, order=1, prefilter=False)

        # setup the a mask for the FOV that can be reconstructed (inner circle)
        x = np.linspace(-self.num_pix / 2 + 0.5, self.num_pix / 2 - 0.5,
                        self.num_pix)
        X0, X1 = np.meshgrid(x, x, indexing='ij')
        R = np.sqrt(X0**2 + X1**2)
        self.mask = (R <= x.max())

        # setup the subset slices for subset projections
        self.subset_slices = []
        self.subset_projection_angles = []
        self.subset_sino_shapes = []

        shuffled_views = np.zeros(self.num_subsets, dtype=np.int16)

        if self.num_subsets > 1:
            shuffled_views[0::2] = np.arange(0, self.num_subsets // 2)
            shuffled_views[1::2] = np.arange(self.num_subsets // 2,
                                             self.num_subsets)

        for i, v in enumerate(shuffled_views):
            self.subset_slices.append(
                (slice(v, None, num_subsets), slice(None, None, None)))
            self.subset_projection_angles.append(
                self.projection_angles[self.subset_slices[i][0]])
            self.subset_sino_shapes.append(
                (self.subset_projection_angles[i].shape[0], self.num_pix))

    def forward_subset(self, img: np.ndarray, subset: int) -> np.ndarray:
        sino = np.zeros(self.subset_sino_shapes[subset])

        for i, angle in enumerate(self.subset_projection_angles[subset]):
            sino[i, ...] = rotate(img, angle,
                                  **self.rotation_kwargs).sum(axis=0)

        return sino * self.pix_size_mm

    def forward(self, img: np.ndarray) -> np.ndarray:
        sino = np.zeros(self.sino_shape)
        for i, sl in enumerate(self.subset_slices):
            sino[sl] = self.forward_subset(img, i)

        return sino

    def adjoint_subset(self, sino: np.ndarray, subset: int) -> np.ndarray:
        img = np.zeros(self.img_shape)

        for i, angle in enumerate(self.subset_projection_angles[subset]):
            tmp_img = np.tile(sino[i, :], self.num_pix).reshape(self.img_shape)
            img += rotate(tmp_img, -angle, **self.rotation_kwargs)

        return img * self.pix_size_mm

    def adjoint(self, sino: np.ndarray) -> np.ndarray:
        img = np.zeros(self.img_shape)

        for i, sl in enumerate(self.subset_slices):
            img += self.adjoint_subset(sino[sl], i)

        return img


#---------------------------------------------------------------------------------------------------


class PETAcquisitionModel(LinearSubsetOperator):
    """ PET data acquisition model

    Parameters
    ----------

    proj ... forward projector

    attenuation_img ... np.ndarray
                        the attenuation image, unit [1/mm]

    res_FWHM_mm     ... float
                        FWHM of Gaussian kernel [mm] used to model the resolution

    sensitivity     ... float
                        scalar sensivity of acquisitiion system

    contamination   ... float
                        scalar contamination added to linear forward model
    """
    def __init__(self,
                 proj: RotationBased2DProjector,
                 attenuation_img: np.ndarray,
                 res_FWHM_mm: float = 5.5,
                 contamination: float = 1e-3,
                 sensitivity: float = 1.) -> None:

        self.proj = proj
        self.attenuation_img = attenuation_img

        # calculate the attenuation sinogram
        self.attenuation_sino = np.exp(
            -self.proj.forward(self.attenuation_img))

        # calculate sigma of the gaussian kernel for resolution modeling
        self.res_sig_mm = res_FWHM_mm / (2.35 * self.proj.pix_size_mm)

        # expectatation of flat contamination
        self.contamination = contamination

        # sensitivity
        self.sensitivity = sensitivity

    def forward_subset(self, img: np.ndarray, subset: int) -> np.ndarray:
        sm_img = gaussian_filter(img, self.res_sig_mm)
        return self.sensitivity * self.attenuation_sino[
            self.proj.subset_slices[subset]] * self.proj.forward_subset(
                sm_img, subset) + self.contamination

    def forward(self, img: np.ndarray) -> np.ndarray:
        sino = np.zeros(self.proj.sino_shape)
        for i, sl in enumerate(self.proj.subset_slices):
            sino[sl] = self.forward_subset(img, i)

        return sino

    def adjoint_subset(self, sino: np.ndarray, subset: int) -> np.ndarray:
        back_img = self.proj.adjoint_subset(
            self.sensitivity *
            self.attenuation_sino[self.proj.subset_slices[subset]] * sino,
            subset)
        return gaussian_filter(back_img, self.res_sig_mm)

    def adjoint(self, sino: np.ndarray) -> np.ndarray:
        img = np.zeros(self.proj.img_shape)

        for i, sl in enumerate(self.proj.subset_slices):
            img += self.adjoint_subset(sino[sl], i)

        return img

    def init_image(self) -> np.ndarray:
        img = np.zeros(self.proj.img_shape)
        img[np.where(self.proj.mask)] = 1

        return img


#-----------------------------------------------------------------------------------------------


class OS_MLEM:
    def __init__(self, emission_sinogram: np.ndarray,
                 acquisition_model: PETAcquisitionModel) -> None:

        self.emission_sinogram = emission_sinogram
        self.acquisition_model = acquisition_model

        # calculate the sensivity images
        self.sensitivity_imgs = []

        for subset in range(self.acquisition_model.proj.num_subsets):
            ones = np.ones(
                self.acquisition_model.proj.subset_sino_shapes[subset])
            tmp = self.acquisition_model.adjoint_subset(ones, subset)
            self.sensitivity_imgs.append(tmp)

        # initialize the images
        self.initialize_image()

        # calculate the FOV indicies
        self.fov_inds = np.where(self.acquisition_model.proj.mask)

    def initialize_image(self) -> None:
        self.image = self.acquisition_model.proj.mask.astype(np.float64)
        self.iteration = 0

    def run_update(self, subset: int) -> None:
        expectation = self.acquisition_model.forward_subset(self.image, subset)

        ratio = self.emission_sinogram[
            self.acquisition_model.proj.subset_slices[subset]] / expectation

        self.image[self.fov_inds] *= self.acquisition_model.adjoint_subset(
            ratio, subset)[self.fov_inds]
        self.image[self.fov_inds] /= self.sensitivity_imgs[subset][
            self.fov_inds]

    def run(self,
            num_iter: int,
            initialize_image: bool = True,
            verbose: bool = False) -> np.ndarray:
        if initialize_image:
            self.initialize_image()

        for i in range(num_iter):
            for subset in range(self.acquisition_model.proj.num_subsets):
                self.run_update(subset)
                if verbose:
                    print(f'iteration {(i+1):03} subset {subset:03}', end='\r')

        return self.image
