# This code is copied from https://github.com/jacobjma/fourier-scale-calibration/blob/main/fourier_scale_calibration/fourier_scale_calibration.py
# Describe what have benn changed
import numpy as np
from scipy import ndimage
from simulate import superpose_deltas

from numba import njit
@njit(cache=True)
def fast_rotate(points, angle_rad, center):
    """Numba-optimized rotation of points about a center."""
    # Precompute trigonometric functions
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Pre-center calculations
    cx, cy = center[0], center[1]
    
    # Initialize output array
    rotated_points = np.empty_like(points)
    
    # Vectorized rotation calculation
    for i in range(points.shape[0]):
        # Translate point to origin
        x = points[i, 0] - cx
        y = points[i, 1] - cy
        
        # Apply rotation
        rotated_x = x * cos_theta - y * sin_theta
        rotated_y = x * sin_theta + y * cos_theta
        
        # Translate back and store
        rotated_points[i, 0] = rotated_x + cx
        rotated_points[i, 1] = rotated_y + cy
    
    return rotated_points

def rotate(points, angle, center=None):
    """Original interface wrapper for Numba-optimized rotation."""
    if center is None:
        center = np.array([0., 0.])
    return fast_rotate(points, np.deg2rad(angle), center)


def regular_polygon(side_length, num_sides):
    """
    Calculate the vertices of a regular polygon given the side length and number of sides.

    Parameters:
    side_length (float): The length of each side of the polygon.
    num_sides (int): The number of sides (or vertices) of the polygon.

    Returns:
    np.ndarray: An array of shape (num_sides, 2) containing the (x, y) coordinates of the polygon's vertices.
    """

    # Initialize an array to store the coordinates of the vertices
    vertices = np.zeros((num_sides, 2), dtype=np.float32)
    # Create an array of indices from 0 to num_sides-1
    indices = np.arange(num_sides, dtype=np.int32)  

    # Calculate the radius of the circumcircle of the polygon
    radius = side_length / (2 * np.sin(np.pi / num_sides)) 

    # Calculate the angle for each vertex with respect to the positive x axes
    angles = 2 * np.pi * indices / num_sides

    # Compute the x and y coordinates using the radius and angles
    vertices[:, 0] = radius * np.sin(angles)
    vertices[:, 1] = radius * np.cos(angles)

    return vertices


def cosine_window(x, cutoff, rolloff):
    rolloff *= cutoff
    array = .5 * (1 + np.cos(np.pi * (x - cutoff + rolloff) / rolloff))
    array[x > cutoff] = 0.
    array = np.where(x > cutoff - rolloff, array, np.ones_like(x))
    return array



def specific_rectangle_crop(image, x, y, pxl_width=256):
    """
    Crops a square region from the image starting at (x, y) with width z.

    Parameters:
    - image: Input image (numpy array).
    - x, y: Top-left coordinates of the crop.
    - z: Width (height) of the square crop.

    Returns:
    - Cropped square image.
    """
    height, width = image.shape[:2]  # Get image dimensions

    # Ensure cropping region stays within image boundaries
    x_end = min(x + pxl_width, width)  
    y_end = min(y + pxl_width, height)

    return image[y:y_end, x:x_end]  # Crop the square region



def square_crop(image):
    # Get the shape of the input image
    height, width = image.shape[:2]

    # If height is greater than width, crop the height
    if height > width:
        n = (height - width) // 2
        return image[n:n + width, ...]
    # If width is greater than height, crop the width
    elif width > height:
        n = (width - height) // 2
        return image[:, n:n + height, ...]
    # If height and width are equal, return the image as is
    else:
        return image


def windowed_fft(image, cf=0.5, rf=0.33):
    image = square_crop(image)
    x = np.fft.fftshift(np.fft.fftfreq(image.shape[-2]))
    y = np.fft.fftshift(np.fft.fftfreq(image.shape[-1]))
    r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)
    m = cosine_window(r, cutoff=cf, rolloff=rf)
    f = np.fft.fft2(image * m)
    return f


def periodic_smooth_decomposition(I):
    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)

    s = v2s(v_fft)

    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f
    return p, s_f


def u2v(u):
    v = np.zeros(u.shape, dtype=np.float64)

    v[0, :] = np.subtract(u[-1, :], u[0, :])
    v[-1, :] = np.subtract(u[0, :], u[-1, :])

    v[:, 0] += np.subtract(u[:, -1], u[:, 0])
    v[:, -1] += np.subtract(u[:, 0], u[:, -1])
    return v


def v2s(v_hat):
    M, N = v_hat.shape

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2 * np.cos(np.divide((2 * np.pi * q), M)) + 2 * np.cos(np.divide((2 * np.pi * r), N)) - 4)

    s = np.zeros_like(v_hat)
    s[den != 0] = v_hat[den != 0] / den[den != 0]
    s[0, 0] = 0
    return s




def detect_fourier_spots(image, template, symmetry, min_scale=None, max_scale=None, nbins_angular=None,
                         return_positions=False, normalize_radial=False, normalize_azimuthal=False,
                         ps_decomp=True):
    if symmetry < 2:
        raise RuntimeError('symmetry must be 2 or greater')

    max_max_scale = (min(image.shape[-2:]) // 2) / np.max(np.linalg.norm(template, axis=1))

    if min_scale is None:
        min_scale = 1 / np.min(np.linalg.norm(template, axis=1))

    if max_scale is None:
        max_scale = max_max_scale

    else:
        max_scale = min(max_scale, max_max_scale)

    if min_scale > max_scale:
        raise RuntimeError('min_scale must be less than max_scale')

    if nbins_angular is None:
        nbins_angular = int(np.ceil((2 * np.pi / symmetry) / 0.01))

    if ps_decomp:
        if len(image.shape) == 3:
            image, _ = np.mean([periodic_smooth_decomposition(i) for i in image], axis=0)
        else:
            image, _ = periodic_smooth_decomposition(image)

    f = np.abs(np.fft.fft2(image))

    if len(f.shape) == 3:
        f = f.mean(0)

    f = np.fft.fftshift(f)

    angles = np.linspace(0, 2 * np.pi / symmetry, nbins_angular, endpoint=False)
    scales = np.arange(min_scale, max_scale, 1)

    r = np.linalg.norm(template, axis=1)[:, None, None] * scales[None, :, None]
    a = np.arctan2(template[:, 1], template[:, 0])[:, None, None] + angles[None, None, :]

    templates = np.array([(np.cos(a) * r).ravel(), (np.sin(a) * r).ravel()])
    templates += np.array([f.shape[0] // 2, f.shape[1] // 2])[:, None]

    unrolled = ndimage.map_coordinates(f, templates, order=1)
    unrolled = unrolled.reshape((len(template), len(scales), len(angles)))

    unrolled = (unrolled).mean(0)

    if normalize_azimuthal:
        unrolled = unrolled / unrolled.mean((1,), keepdims=True)

    if normalize_radial:
        unrolled = unrolled / unrolled.mean((0,), keepdims=True)

    p = np.unravel_index(np.argmax(unrolled), unrolled.shape)

    if return_positions:
        r = np.linalg.norm(template, axis=1) * scales[p[0]]  # + min_scale
        a = np.arctan2(template[:, 1], template[:, 0])
        a -= p[1] * 2 * np.pi / symmetry / nbins_angular + np.pi / symmetry

        spots = np.array([(np.cos(a) * r).ravel(), (np.sin(a) * r).ravel()]).T
        spots += np.array([f.shape[0] // 2, f.shape[1] // 2])[None]
        return scales[p[0]], spots
    else:
        return scales[p[0]]


class FourierSpaceCalibrator:

    def __init__(self, template, lattice_constant, min_sampling=None, max_sampling=None,
                 normalize_radial=False, normalize_azimuthal=True, angle=30, layer_angles=None, fft_order=None):
        self.template = template    # The vertices of a polygon that will be used as a template using the function regular_polygon
        self.lattice_constant = lattice_constant
        self.min_sampling = min_sampling
        self.max_sampling = max_sampling
        self.normalize_radial = normalize_radial
        self.normalize_azimuthal = normalize_azimuthal
        self._spots = None
        self.angle = angle
        self.layer_angles = layer_angles or []
        self.fft_order = fft_order or 1
    def get_spots(self):
        return self._spots

    def get_mask(self, shape, sigma):
        array = np.zeros(shape)
        spots = self.get_spots()[:, ::-1]
        superpose_deltas(spots, 0, array[None])
        array = np.fft.fftshift(array)
        x = np.fft.fftfreq(shape[0])
        y = np.fft.fftfreq(shape[1])
        z = np.exp(-(x[:, None] ** 2 + y[None] ** 2) * sigma ** 2 * 4)
        array = np.fft.ifft2(np.fft.fft2(array) * z).real
        return array

    def fourier_filter(self, image, sigma):
        spots = self.get_mask(image.shape, 4)
        spots /= spots.max()
        return np.fft.ifft2(np.fft.fft2(image) * spots).real

    def calibrate(self, image, return_spots=False):

        """03-04-2025 Added new templates for creating graphene multilayers. The initial idea is to switch to a more general form by having
        a slider for the number of layers and slider of the fft order, but this failed to work (need another try), so I kept hexagonal and 2nd-order-hexagonal untouched. 
        One need to add a slider for the number of layers and a slider for the angles between the layers."""
        if self.template.lower() == 'hexagonal':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            symmetry = 6
        elif self.template.lower() == '2nd-order-hexagonal':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            template = np.vstack((template, rotate(template, 30) * np.sqrt(3)))
            symmetry = 6

        elif self.template.lower().startswith('graphene-'):
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3) 
            layers = int(self.template.split('-')[1][0])
            base = regular_polygon(1., 6)
            if self.fft_order == 2:
                base = np.vstack((base, rotate(base, 30) * np.sqrt(3)))
            template = base.copy()
            
            if self.layer_angles and len(self.layer_angles) >= (layers - 1):
                for angle in self.layer_angles[:layers-1]:
                    rotated = rotate(base, angle)
                    template = np.vstack((template, rotated))
            else:
                theta = 30.0 / layers
                for i in range(1, layers):
                    rotated = rotate(base, i*theta)
                    template = np.vstack((template, rotated))
            
            symmetry = 6


        elif self.template.lower() == 'ring':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            sidelength = 2 * np.sin(np.pi / 128)
            template = regular_polygon(sidelength, 128)
            symmetry = 128
        else:
            raise NotImplementedError()

        if self.min_sampling is None:
            min_scale = None
        else:
            min_scale = k * self.min_sampling

        if self.max_sampling is None:
            max_scale = None
        else:
            max_scale = k * self.max_sampling

        scale, spots = detect_fourier_spots(image, template, symmetry, min_scale=min_scale, max_scale=max_scale,
                                            return_positions=True, normalize_azimuthal=self.normalize_azimuthal,
                                            normalize_radial=self.normalize_radial)

        self._spots = spots
        if return_spots:
            return scale / k, spots
        else:
            return scale / k

    def __call__(self, image):
        return self.calibrate(image)