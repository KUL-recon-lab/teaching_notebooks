import abc
import numpy as np
import math

import scipy.ndimage as ndi

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

#-------------------------------------------------------------------------

class SubsetSlicer(abc.ABC):
    """ abstract base class for defines subset slices """
    def __init__(self, complete_shape: tuple[int,...], num_subsets: int):

        self._complete_shape = complete_shape
        self._num_subsets = num_subsets
        self._ndim = len(self._complete_shape)

    @property
    def num_subsets(self) -> int:
        return self._num_subsets

    @abc.abstractmethod
    def get_subset_slice(self, subset: int) -> tuple[slice, ...]:
        """ get the slice of a subset """

    @abc.abstractmethod
    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        """ get the shape of a subset """

#-------------------------------------------------------------------------

class InterleavedSubsetSlicer(SubsetSlicer):
    """ class for interleaved subsets along a given axix """
    def __init__(self, 
                 complete_shape: tuple[int,...], 
                 num_subsets: int,
                 subset_axis: int):

        super().__init__(complete_shape, num_subsets)

        self._subset_axis = subset_axis

        self.init_subsets()

    def init_subsets(self) -> None:
        self._subset_slices = []
        self._subset_shapes = []

        all_views = np.zeros(self._num_subsets, dtype=np.int16)

        # interleave the views to maximize the distance between the subsets
        if self._num_subsets > 1:
            all_views[0::2] = np.arange(0, math.ceil(self._num_subsets/2))
            all_views[1::2] = np.arange(math.ceil(self._num_subsets/2),
                                             self._num_subsets)

        empty_slice = self._ndim * [slice(None, None, None)]

        for i, v in enumerate(all_views):
            sl = empty_slice.copy()
            sl[self._subset_axis] = slice(v, None, self._num_subsets)

            self._subset_slices.append(tuple(sl))

            sh = list(self._complete_shape)
            sh[self._subset_axis] = math.ceil((self._complete_shape[self._subset_axis] - v) / self._num_subsets)
            self._subset_shapes.append(tuple(sh))


    def get_subset_slice(self, subset: int) -> tuple[slice, ...]:
        """ get the slice of a subset """
        return self._subset_slices[subset]

    def get_subset_shape(self, subset: int) -> tuple[int, ...]:
        """ get the shape of a subset """
        return self._subset_shapes[subset]


#-------------------------------------------------------------------------

class LinearSubsetOperator(abc.ABC):
    """ abstract base class for linear subset operator """
    def __init__(self, 
                 x_shape: tuple[int, ...], 
                 y_shape: tuple[int, ...], 
                 subset_slicer: SubsetSlicer):

        self._x_shape = x_shape
        self._y_shape = y_shape
        self.subset_slicer = subset_slicer

    @property
    def x_shape(self) -> tuple[int, ...]:
        return self._x_shape

    @property
    def y_shape(self) -> tuple[int, ...]:
        return self._y_shape

    @abc.abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """ forward step """

    @abc.abstractmethod
    def forward_subset(self, x: np.ndarray, subset: int) -> np.ndarray:
        """ subset forward step """

    @abc.abstractmethod
    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """ adjoint of forward step """

    @abc.abstractmethod
    def adjoint_subset(self, y: np.ndarray, subset: int) -> np.ndarray:
        """ adjoint of subset forward step """

    def test_subset_adjoint(self, subset: int, rtol: float = 1e-3) -> None:
        x = np.random.rand(*self.x_shape)
        y = np.random.rand(*self.subset_slicer.get_subset_shape(subset))

        x_fwd  = self.forward_subset(x, subset)
        y_back = self.adjoint_subset(y, subset)

        return np.isclose((x_fwd*y).sum(),  (y_back*x).sum(), rtol = rtol)

#-------------------------------------------------------------------------

class Projector(LinearSubsetOperator):
    """ base class for line / volume integral projectors """

    def __init__(self, 
                 image_shape: tuple[int, ...], 
                 data_shape: tuple[int, ...],
                 voxel_size_mm: tuple[float, ...],
                 subset_slicer: SubsetSlicer):

        super().__init__(image_shape, data_shape, subset_slicer)

        self._voxel_size_mm = voxel_size_mm

    @property
    def voxel_size_mm(self) -> tuple[float, ...]:
        return self._voxel_size_mm

    def forward(self, x: np.ndarray) -> np.ndarray:
        """ forward step """
        raise NotImplementedError

    def forward_subset(self, x: np.ndarray, subset: int) -> np.ndarray:
        """ subset forward step """
        raise NotImplementedError

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """ adjoint of forward step """
        raise NotImplementedError

    def adjoint_subset(self, y: np.ndarray, subset: int) -> np.ndarray:
        """ adjoint of subset forward step """
        raise NotImplementedError


#-------------------------------------------------------------------------

class RotationBased2DProjector(Projector):
    """ base class for line / volume integral projectors """

    def __init__(self, 
                 num_pixel: int, 
                 pixel_size_mm: float,
                 num_subsets: int = 1,
                 projection_angles: np.ndarray = None):

        if projection_angles is None:
            self._projection_angles = np.linspace(0, 180, 180, endpoint = False)
        else:
            self._projection_angles = projection_angles

        sinogram_shape = (self._projection_angles.shape[0], num_pixel)

        super().__init__((num_pixel, num_pixel), 
                         sinogram_shape, 
                         (pixel_size_mm, pixel_size_mm), 
                         InterleavedSubsetSlicer(sinogram_shape, num_subsets, 0))

        self.rotation_kwargs = dict(reshape=False, order=1, prefilter=False)

        # setup the a mask for the FOV that can be reconstructed (inner circle)
        x = np.linspace(-num_pixel / 2 + 0.5, num_pixel / 2 - 0.5, num_pixel)
        X0, X1 = np.meshgrid(x, x, indexing='ij')
        R = np.sqrt(X0**2 + X1**2)
        self.mask = (R <= x.max()).astype(float)


    def forward(self, image: np.ndarray) -> np.ndarray:
        """ forward step """

        sino = np.zeros(self.y_shape)
        for subset in range(self.subset_slicer.num_subsets):
            subset_slice = self.subset_slicer.get_subset_slice(subset)
            sino[subset_slice] = self.forward_subset(image, subset)

        return sino

    def forward_subset(self, image: np.ndarray, subset: int) -> np.ndarray:
        """ subset forward step """

        subset_shape = self.subset_slicer.get_subset_shape(subset)
        sino = np.zeros(subset_shape)

        subset_slice = self.subset_slicer.get_subset_slice(subset)

        masked_image = self.mask * image

        for i, angle in enumerate(self._projection_angles[subset_slice[0]]):
            sino[i, ...] = ndi.rotate(masked_image, angle,
                                  **self.rotation_kwargs).sum(axis=0)

        return sino * self.voxel_size_mm[0]

    def adjoint(self, sinogram: np.ndarray) -> np.ndarray:
        """ adjoint of forward step """
        image = np.zeros(self.x_shape)

        for subset in range(self.subset_slicer.num_subsets):
            subset_slice = self.subset_slicer.get_subset_slice(subset)
            image += self.adjoint_subset(sinogram[subset_slice], subset)

        return image


    def adjoint_subset(self, sinogram: np.ndarray, subset: int) -> np.ndarray:
        image = np.zeros(self.x_shape)

        subset_slice = self.subset_slicer.get_subset_slice(subset)

        for i, angle in enumerate(self._projection_angles[subset_slice[0]]):
            tmp_img = np.tile(sinogram[i, :], self.x_shape[0]).reshape(self.x_shape)
            image += ndi.rotate(tmp_img, -angle, **self.rotation_kwargs)

        return image * self.mask * self.voxel_size_mm[0]




#-------------------------------------------------------------------------

if __name__ == "__main__":

    num_pix = 150
    em_img, att_img = test_images(num_pix)

    proj = RotationBased2DProjector(num_pix, 3., 20)

    q = proj.forward(em_img)
    w = proj.adjoint(q)
