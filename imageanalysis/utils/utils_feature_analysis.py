"""
Utility functions for feature analysis in image processing.
Author: Somar Dibeh
Date: 2025-07-15
"""

import cv2
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter, gaussian_filter


DEFAULTS = {
    "Gamma": 1.0,
    "CLAHE_Clip": 1.0,
    "CLAHE_Tile": 8,
    "Gaussian_Sigma": 1.0,
    "Percentile Contrast": 1.0,
    "Double_Gaussian_Sigma1": 0.5,
    "Double_Gaussian_Sigma2": 0.2,
    "Double_Gaussian_Weight": 0.5,
    "Dilation": 0,
    "Erosion": 0,
    "Opening": 0,
    "Closing": 0,
    "Gradient": 0,
    "Boundary": 0,
    "Circularity": 0.05,
    "Opening2": 0,
    "Closing2": 0,
    "Dilation2": 0,
    "Erosion2": 0,
    "Gradient2": 0,
    "Boundary2": 0,
    "Isolation Distance": 0.05,
    "Brightness": 0.0,
    "Sigmoid Alpha": 10.0,
    "Sigmoid Beta": 0.5,
    "Exp Transform": False,
    "Log Transform": False,
    "Contour_Retrieval_Mode": "RETR_EXTERNAL",
    "Contour_Approximation_Method": "CHAIN_APPROX_TC89_KCOS",
    "Analysis_Type": "Clean_area_analysis",
    "Feature_Analysis_Type": "Single_atom_clusters_analysis",
    # "Resize_Factor": 1.0,
    # "Resize_Method": "Bicubic",
    "Kernel_Size": 3,
    "Kmeans_Initialization": "K-means++",
    "Kmeans_Clusters_Number": 2,
    "Kmeans_Attempts_Number": 10,
    "Kmeans_Epsilon": 0.01,
    "Threshold_Method": "manual",
}


COLUMNS_ORDER = ['Slice',  'Analysis_Type', 'Number_of_Layers','Feature_Analysis_Type', 'Contour_retrieval_modes', 'Contour_approximation_methods',
                 'Threshold1', 'Threshold2', 'Threshold_Method', 'Threshold_SA1','Threshold_SA2', 'Kmeans_Initialization',
                 'Thresholds_List_Kmeans','Kmeans_Clusters_Number', 'Kmeans_Attempts_Number', 'Kmeans_Epsilon', 
                #  'Resize_Factor', 'Resize_Method',
                 'Clean_Area_nm2', 'Contamination_Area_nm2', 
                 'Number_of_Clusters', 'Total_Cluster_Area_nm2', 'Clusters_Density', 'Clusters', 'Num_Atoms', 
                 'Atoms_Area', 'Atoms_Density', 'Atoms', 'Calibrated_FOV', 'Entire_Area_nm2', 
                 'Circularities', 'Roundness', 'Feret_Diameter', 'Aspect_Ratio', 
                 'Min_Clean_Cont_Area_nm2', 'Max_Clean_Cont_Area_nm2', 'Min_Cluster_Area_nm2', 'Max_Cluster_Area_nm2',
                 'Circularity', 'Isolation Distance','Make Circular threshold', 'SA_Cluster_Definer', 'Percentile Contrast',
                 'Kernel_Size', 'Gaussian_Sigma','Double_Gaussian_Sigma1', 'Double_Gaussian_Sigma2', 'Double_Gaussian_Weight',
                 "Histogram_Peaks",
                 'Dilation', 'Erosion', 'Opening', 'Closing', 'Gradient', 'Boundary', 
                 'Opening2','Closing2', 'Dilation2', 'Erosion2', 'Gradient2', 'Boundary2',
                 'Brightness', 'Gamma', 'CLAHE_Clip','CLAHE_Tile',
                 'Sigmoid Alpha', 'Sigmoid Beta', 'Exp Transform', 'Log Transform',]

def contour_min_distance(cnt1, cnt2):
    """Optimized distance calculation with adaptive point sampling"""
    if cnt1 is cnt2:
        return float('inf')
    
    min_dist = float('inf')
    
    # Determine sampling rate based on contour size
    def get_step(contour):
        length = len(contour)
        if length <= 20:    # Check all points for small contours
            return 1
        elif length <= 50:  # Check every 3rd point
            return 4
        else:               # Check every 5th point for large contours
            return 8
    
    # Check points from cnt1 to cnt2 with adaptive sampling
    step1 = get_step(cnt1)
    for pt in cnt1[::step1, 0, :]:
        x, y = int(pt[0]), int(pt[1])
        dist = cv2.pointPolygonTest(cnt2, (x, y), True)
        if dist < 0:
            min_dist = min(min_dist, abs(dist))
        else:
            return 0.0  # Early exit if contours overlap
    
    # Check points from cnt2 to cnt1 with adaptive sampling
    step2 = get_step(cnt2)
    for pt in cnt2[::step2, 0, :]:
        x, y = int(pt[0]), int(pt[1])
        dist = cv2.pointPolygonTest(cnt1, (x, y), True)
        if dist < 0:
            min_dist = min(min_dist, abs(dist))
        else:
            return 0.0  # Early exit if contours overlap
    
    return min_dist


def process_pair( pair, contours, centers, radii, nm_per_pixel, isolation_px, min_dist_func):
    i, j = pair

    area_i = cv2.contourArea(contours[i]) * nm_per_pixel**2
    area_j = cv2.contourArea(contours[j]) * nm_per_pixel**2
    if area_i < 0.025 or area_j < 0.025:    # exclude very small contours
        return None

    center_dist = np.linalg.norm(centers[i] - centers[j])
    # exclude contours which are too far from each others by checking the distance between their centers compared to the radii of the minimum enclosing circles around them
    if (center_dist - (radii[i] + radii[j])) > isolation_px:   
        return None

    dist = min_dist_func(contours[i], contours[j])
    if dist <= isolation_px:
        return (i, j)
    return None


def measure_circularity( contour):
    """Calculate circularity of a contour (4π*Area/Perimeter²)"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def measure_roundness(contour):
    """Calculate roundness (minor axis/major axis) using fitted ellipse"""
    if len(contour) < 5:  # Need at least 5 points to fit ellipse
        return 0.0
    (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
    if major_axis == 0:
        return 0.0
    return minor_axis / major_axis


def measure_aspect_ratio(contour):
    """Calculate aspect ratio (width/height) using bounding rectangle"""
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return 0.0
    return float(w) / h


def measure_feret_diameter(contour, nm_px):
    from itertools import combinations
    """Optimized Feret diameter using convex hull"""
    hull = cv2.convexHull(contour)
    if len(hull) < 2:
        return 0.0
    max_dist = 0.0
    for (p1, p2) in combinations(hull, 2):
        dx = p1[0][0] - p2[0][0]
        dy = p1[0][1] - p2[0][1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > max_dist:
            max_dist = dist
    return max_dist * nm_px 


def compute_shape_metrics(args):
    contour, nm_per_pixel, measure_circularity, measure_roundness, measure_feret_diameter, measure_aspect_ratio = args

    return {
        'circularity': measure_circularity(contour),
        'roundness': measure_roundness(contour),
        'feret_diameter': measure_feret_diameter(contour, nm_per_pixel),
        'aspect_ratio': measure_aspect_ratio(contour)
    }


def compute_area(contour, nm2_per_pixel2):
    return cv2.contourArea(contour) * nm2_per_pixel2



################ Need testing later maybe on the cluster, but performance wise it failed terribly even when tried with multiprocessing, needs more computational power#########


# -----------------------------
# Multi-Gaussian model (shared σ, offset, optional θ)
# -----------------------------
def multi_gaussian2d_shared_sigma(xy, params, shared_theta=False):
    X, Y = xy
    idx = 0
    sx = params[idx]; idx+=1
    sy = params[idx]; idx+=1
    if shared_theta:
        theta = params[idx]; idx+=1
    else:
        theta = 0.0
    offset = params[idx]; idx+=1

    img = np.zeros_like(X, dtype=float)
    ct, st = np.cos(theta), np.sin(theta)
    a = (ct**2)/(2*sx**2) + (st**2)/(2*sy**2)
    b = -st*ct/(2*sx**2) + st*ct/(2*sy**2)
    c = (st**2)/(2*sx**2) + (ct**2)/(2*sy**2)
    while idx < len(params):
        amp = params[idx]; x0 = params[idx+1]; y0 = params[idx+2]; idx += 3
        img += amp * np.exp(-(a*(X-x0)**2 + 2*b*(X-x0)*(Y-y0) + c*(Y-y0)**2))
    return (img + offset).ravel()

# -----------------------------
# Peak finding (local maxima)
# -----------------------------
def find_peaks_2d(img, min_distance=3, threshold_abs=None, footprint_size=3):
    I = np.asarray(img, dtype=float)
    if threshold_abs is None:
        threshold_abs = I.mean() + 2*I.std()
    Is = gaussian_filter(I, sigma=1.0)
    neighborhood = maximum_filter(Is, size=footprint_size)
    peaks_mask = (Is == neighborhood) & (Is > threshold_abs)
    ys, xs = np.nonzero(peaks_mask)
    vals = Is[ys, xs]
    order = np.argsort(-vals)
    selected = []
    for idx in order:
        y0, x0 = ys[idx], xs[idx]
        if all((y0 - y1)**2 + (x0 - x1)**2 >= min_distance**2 for (y1, x1) in selected):
            selected.append((y0, x0))
    return selected

# -----------------------------
# Cluster fitting with AIC/BIC model selection
# -----------------------------
def fit_cluster_auto(roi, max_atoms=6, shared_theta=False, sigma_init_px=2.0):
    I = np.asarray(roi, dtype=float)
    nrows, ncols = I.shape
    y = np.arange(nrows); x = np.arange(ncols)
    X, Y = np.meshgrid(x, y)
    best_model = None

    # Candidate centers from local maxima
    all_peaks = find_peaks_2d(I, min_distance=3)
    if len(all_peaks) == 0:
        return None

    # Try K=1..max_atoms
    for K in range(1, min(max_atoms, len(all_peaks)) + 1):
        centers = all_peaks[:K]
        # initial params
        sx0 = sy0 = float(sigma_init_px)
        offset0 = float(np.percentile(I, 5))
        amps0 = []
        for (yy, xx) in centers:
            amps0 += [max(1e-3, I[int(yy), int(xx)] - offset0), float(xx), float(yy)]
        if shared_theta:
            p0 = [sx0, sy0, 0.0, offset0] + amps0
        else:
            p0 = [sx0, sy0, offset0] + amps0

        try:
            popt, pcov = curve_fit(
                lambda xy, *p: multi_gaussian2d_shared_sigma(xy, p, shared_theta=shared_theta),
                (X, Y), I.ravel(), p0=p0, maxfev=20000
            )
        except Exception:
            continue

        model_img = multi_gaussian2d_shared_sigma((X, Y), popt, shared_theta=shared_theta).reshape(I.shape)
        resid = I - model_img
        rss = np.sum(resid**2)
        n = I.size
        k = len(popt)
        aic = n * np.log(rss/n + 1e-12) + 2*k
        bic = n * np.log(rss/n + 1e-12) + k*np.log(n)

        if best_model is None or bic < best_model["bic"]:
            best_model = {
                "K": K, "popt": popt, "rss": rss, "aic": aic, "bic": bic,
                "centers": centers
            }
    return best_model

# -----------------------------
# Area estimation from Gaussian fit
# -----------------------------
def gaussian_area_from_fit(popt, alpha=0.5, shared_theta=False):
    idx = 0
    sx = popt[idx]; idx+=1
    sy = popt[idx]; idx+=1
    if shared_theta:
        theta = popt[idx]; idx+=1
    else:
        theta = 0.0
    offset = popt[idx]; idx+=1

    atoms = []
    while idx < len(popt):
        amp = popt[idx]; x0 = popt[idx+1]; y0 = popt[idx+2]; idx += 3
        atoms.append((amp, x0, y0))

    # area at alpha-isophote (in pixels²)
    A_alpha = 2*np.pi*sx*sy*np.log(1/alpha)

    return A_alpha, atoms, sx, sy

# -----------------------------
# Drop-in replacement for cv2.contourArea
# -----------------------------
def gaussianContourArea(img, contour, pad_factor=1.1, alpha=0.5):
    # bounding box with padding (Feret diameter proxy)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    feret_d = max(rect[1])  # Feret diameter (long axis)
    cx, cy = rect[0]
    half_size = int(np.ceil((feret_d * pad_factor) / 2))
    y0, x0 = int(cy), int(cx)
    y1, y2 = max(0, y0-half_size), min(img.shape[0], y0+half_size)
    x1, x2 = max(0, x0-half_size), min(img.shape[1], x0+half_size)
    roi = img[y1:y2, x1:x2]

    model = fit_cluster_auto(roi, max_atoms=6)
    if model is None:
        return cv2.contourArea(contour), 0.0  # fallback

    A_alpha, atoms, sx, sy = gaussian_area_from_fit(model["popt"], alpha=alpha)

    # crude uncertainty estimate: scale with residuals
    n = roi.size
    dof = n - len(model["popt"])
    sigma2 = model["rss"]/max(dof, 1)
    rel_err = np.sqrt(2/dof) if dof>0 else 0.5
    A_unc = A_alpha * rel_err

    return A_alpha, A_unc
