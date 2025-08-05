"""
Utility functions for feature analysis in image processing.
Author: Somar Dibeh
Date: 2025-07-15
"""

import cv2
import numpy as np


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
    "Resize_Factor": 1.0,
    "Resize_Method": "Bicubic",
    "Kernel_Size": 3,}


COLUMNS_ORDER = ['Slice',  'Analysis_Type', 'Number_of_Layers','Feature_Analysis_Type', 'Contour_retrieval_modes', 'Contour_approximation_methods',
                 'Threshold1', 'Threshold2', 'Threshold_SA1','Threshold_SA2', 'Resize_Factor', 'Resize_Method','Clean_Area_nm2', 'Contamination_Area_nm2', 
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