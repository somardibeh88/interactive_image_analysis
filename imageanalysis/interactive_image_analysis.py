########
# Title: Interactive Image Analysis
# Author: Somar Dibeh
# Date: 2025-04-25
########

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
import matplotlib
matplotlib.use("Qt5Agg")
import os
import json
import h5py
import cv2
import csv
import joblib
import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.offsetbox import AnchoredText
from matplotlib.font_manager import FontProperties 
import matplotlib.patches as patches
from datetime import datetime
from IPython.display import display, clear_output
from traitlets import link
import time
from ipywidgets import interactive_output, HBox, VBox, FloatSlider, IntSlider, Checkbox, Button, Output, Dropdown, IntText, FloatText, Text
from .data_loader import DataLoader
from .filters import *
from . import fourier_scale_calibration as fsc
from .fourier_scale_calibration import *
from .utils import *
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

cv2.ocl.setUseOpenCL(False)



class InteractiveImageAnalysis():

    materials = {'hBN': 2.504, 'Graphene': 2.46, 'MoS2': 3.212 }
    calibration_types = ['hexagonal','2nd-order-hexagonal', 'graphene-2L','graphene-3L', 'graphene-4L', 'graphene-5L']


    contour_retrieval_modes = { 'RETR_EXTERNAL': cv2.RETR_EXTERNAL,
                                'RETR_CCOMP': cv2.RETR_CCOMP}

    contour_approximation_methods = { 'CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE,
                                      'CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE,
                                      'CHAIN_APPROX_TC89_L1': cv2.CHAIN_APPROX_TC89_L1,
                                      'CHAIN_APPROX_TC89_KCOS': cv2.CHAIN_APPROX_TC89_KCOS}
    
    INTERPOLATION_MAP = {'Bilinear': cv2.INTER_LINEAR,
                         'Bicubic': cv2.INTER_CUBIC,
                         'Lanczos': cv2.INTER_LANCZOS4,
                         'Nearest': cv2.INTER_NEAREST,
                         'Area': cv2.INTER_AREA,}
    # Add FFT order control
    fft_orders = ['1st-order', '2nd-order']
    def __init__(self, stack_path, metadata_path=None, analysing_features=False, save_images_with_calibrated_scalebar=True, 
                 clean_graphene_analysis=True, contamination_analysis=False, fixed_length_scalebar=False, clusters_analysis=False, defects_analysis=False, font_path=None):

        self.stack = DataLoader(stack_path)
        self.metadata = self.stack.raw_metadata

        self.clusters_sa_analysis = clusters_analysis
        self.defects_analysis = defects_analysis
        self.font_path = font_path

        self.define_global_font_matplotlib()
        self.plot_widget = QWidget()
        self.plot_widget.setLayout(QVBoxLayout())
        self.results = None
        self.clean_graphene_analysis = clean_graphene_analysis
        self.contamination_analysis = contamination_analysis
        self.analysing_features = analysing_features
        self.save_images_with_calibrated_scalebar = save_images_with_calibrated_scalebar
        self.calibration_factor = None
        self.ref_fov = None
        self.selected_region = None
        self.need_calibration = True
        self.fixed_position_scalebar = fixed_length_scalebar


        # Thresholding settings : for defining contour retrieval modes and approximation methods
        self.contour_retrieval_dropdown = widgets.Dropdown( options=list(self.contour_retrieval_modes.keys()),
                                                            value='RETR_EXTERNAL',
                                                            description='Retrieval Mode:',
                                                            style={'description_width': '140px'},
                                                            layout={'width': '360px'})

        self.contour_approximation_dropdown = widgets.Dropdown( options=list(self.contour_approximation_methods.keys()),
                                                            value='CHAIN_APPROX_TC89_KCOS',
                                                            description='Approximation:',
                                                            style={'description_width': '140px'},
                                                            layout={'width': '360px'})



        # Colormap
        self.colormap_dropdown = Dropdown(
            options=[ 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
        'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'],
            value='gray',
            description='Colormap:',
            continuous_update=False
        )


        self.cv2_colormaps = {
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'inferno': cv2.COLORMAP_INFERNO,
            'magma': cv2.COLORMAP_MAGMA,
            'cividis': cv2.COLORMAP_CIVIDIS,
            'jet': cv2.COLORMAP_JET,
            'turbo': cv2.COLORMAP_TURBO,
            'hot': cv2.COLORMAP_HOT,
            'cool': cv2.COLORMAP_COOL,
            'spring': cv2.COLORMAP_SPRING,
            'summer': cv2.COLORMAP_SUMMER,
            'autumn': cv2.COLORMAP_AUTUMN,
            'winter': cv2.COLORMAP_WINTER}

        self.line_profile_widgets = self.add_line_profile_widgets()
        self.region_profile_widgets = self.add_region_profile_widgets()

        # Add FOV editing widgets
        self.fov_input = FloatText(value=16.0, description='FOV (nm):', step=0.1, style={'description_width': '120px'}, layout={'width': '380px'})
        self.save_fov_button = Button(description="Save FOV", tooltip="Save FOV to metadata. Careful, this will overwrite your existing metadata file")
        self.save_fov_button.on_click(self.save_fov_to_metadata)

        #################################  Various filters Widgets  #######################################
        self.t = 16
        self.width = 512
        self.kernel_size = 3
        self.kernel = np.ones((self.kernel_size, self.kernel_size), dtype=np.uint8)
        self.kernel_size_slider = IntSlider(min=1, max=15, value=3, step=2, description='Kernel Size',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'}, continuous_update=False)
        self.slice_slider = IntSlider(min=0, max=len(self.stack.raw_data) - 1, value=0, description='Slice', continuous_update=False)
        self.threshold_slider = FloatSlider(min=0, max=255, value=100, step=0.2, description='Threshold', continuose_update=False)
        self.threshold_slider_sa = FloatSlider(min=0, max=255, value=100, step=0.5, description='Threshold_sa', style={'description_width': '120px'}, layout={'width': '320px'}, continuous_update=False)
        self.contrast_slider = FloatSlider(min=0.005, max=5.0, value=1.0, step=0.005, description='Contrast', layout={'display': 'none'}, continuose_update=False)
        self.double_gaussian_slider1 = FloatSlider(min=0.05, max=2.0, value=0.5, step=0.01, description='Double gaussian sigma 1', style={'description_width': '200px'}, layout={'display': 'none', 'width': '400px'}, continuose_update=False)
        self.double_gaussian_slider2 = FloatSlider(min=0.05, max=2.0, value=0.2, step=0.01, description='Double gaussian sigma 2', style={'description_width': '200px'}, layout={'display': 'none', 'width': '400px'}, continuose_update=False)
        self.double_gaussian_weight_slider = FloatSlider(min=0.1, max=1.0, value=0.5, step=0.01, description='Double gaussian weight', style={'description_width': '160px'}, layout={'display': 'none', 'width': '420px'}, continuose_update=False)
        self.gamma_slider = FloatSlider(min=0.02, max=5.0, value=1, step=0.01, description='Gamma', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'},continuous_update=False)
        self.clahe_clip_slider = FloatSlider(min=0.1, max=4.0, value=1, step=0.01, description='CLAHE Clip', layout={'display': 'none'}, continuous_update=False)
        self.clahe_tile_slider = IntSlider(min=1, max=42, value=self.t, step=1, description='CLAHE Tile', layout={'display': 'none'}, continuous_update=False)
        self.gaussian_sigma_slider = FloatSlider(min=0.02, max=8.0, value=1.0, step=0.02, description='Gaussian sigma', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.min_area_slider = FloatSlider(min=0, max=100, value=50, step=0.02, description='Min Clean_Cont Area', style={'description_width': '160px'},layout={'display': 'none', 'width': '360px'})
        self.max_area_slider = FloatSlider(min=0, max=120000, value=10000, step=5, description='Max Clean_Cont Area', style={'description_width': '160px'},layout={'display': 'none', 'width': '360px'})
        self.min_area_sa_clusters = FloatSlider(min=0, max=200, value=0.1, step=0.001, description='Min Clust_sa Area', style={'description_width': '160px'},layout={'display': 'none', 'width': '360px'}, readout_format='.3f')
        self.max_area_sa_clusters = FloatSlider(min=0, max=2000, value=1, step=0.005, description='Max Cluster_sa Area', style={'description_width': '160px'},layout={'display': 'none', 'width': '360px'},readout_format='.3f')
        self.circularity_slider = FloatSlider(min=0, max=1, value=0.5, step=0.01, description='Min Circularity', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'},readout_format='.3f')
        self.min_isolation_slider = FloatSlider(min=0.002, max=100, value=1, step=0.004, description='Min Isolation', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'}, readout_format='.3f')
        self.single_atom_clusters_definer = FloatSlider(min=0, max=2, value=0.5, step=0.005, description='SA Cluster definer', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'}, readout_format='.3f')
        self.make_circular_thresh = FloatSlider(min=0, max=2, value=0.03, step=0.005, description='Make circular thresh', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        # Brightness/Contrast Controls
        self.brightness_slider = FloatSlider(min=-150, max=250, value=0, step=1, description='Brightness', layout={'display': 'none'})
        self.sigmoid_alpha = FloatSlider(min=0, max=20, value=10, step=0.5, description='Sigmoid Alpha', layout={'display': 'none'})
        self.sigmoid_beta = FloatSlider(min=0, max=1, value=0.5, step=0.01, description='Sigmoid Beta', layout={'display': 'none'})

        ################## FFT Calibration widgets  #############################
        self._last_calibration = None
        self._cached_nm_per_pixel = None
        # Filters
        self.gamma_fft_slider = FloatSlider(min=0.1, max=5.0, value=0.5, step=0.01, description='Gamma', style={'description_width': '100px'}, layout={ 'width': '400px'})  
        self.contrast_fft_slider = FloatSlider(min=0.002, max=1.0, value=1.0, step=0.002, description='Contrast', layout={'width': '400px'})
        self.gaussian_sigma_fft_slider = FloatSlider(min=0.01, max=1.2, value=0.4, step=0.02, description='Gaussian sigma', style={'description_width': '100px'}, layout={ 'width': '400px'})
        # # Add new resizing widgets
        self.resize_checkbox = Checkbox(value=False, description='Enable Resizing')
        self.resize_method = Dropdown(options=['Bilinear', 'Bicubic', 'Lanczos', 'Nearest', 'Area'], value='Bicubic', description='Resize Method:', layout={'width': '300px'})
        self.resize_factor = IntSlider(min=1, max=10, value=1, step=1, description='Resize Factor:', style={'description_width': '200px'}, layout={'width': '400px'})

        self.calibrator = None
        self.use_new_image_checkbox = Checkbox(value=False, description='Use New Image for Calibration',
                                                tooltip="Must be checked when changing images to update reference FOV")
        
        self.materials_dropdown = widgets.Dropdown(options=list(self.materials.keys()), value='Graphene', description='Material:')
        self.calibration_type_dropdown = widgets.Dropdown(options=self.calibration_types, value='hexagonal', description='Calibration Type:')
        self.min_sampling_slider = FloatSlider(min=0.005, max=2.0, value=0.05, step=0.005, description='Min Sampling:', style={'description_width': '120px'}, layout={'width': '400px'})
        self.max_sampling_slider = FloatSlider(min=0.01, max=2.0, value=0.8, step=0.01, description='Max Sampling:', style={'description_width': '120px'}, layout={'width': '400px'})
        self.n_widget_sliders = IntSlider(min=1, max=1600, value=200, step=1, description='N Sliders:', style={'description_width': '120px'}, layout={'width': '400px'})
        self.fft_spots_rotation_slider = FloatSlider(min=0, max=360, value=0, step=1, description='Rotation:', style={'description_width': '120px'}, layout={'width': '400px'})
        self.rolloff_slider = FloatSlider(min=0.05, max=2, value=0.33, step=0.05, description='Rolloff:', style={'description_width': '120px'}, layout={'width': '400px'})
        self.cuttoff_slider = FloatSlider(min=0.05, max=2, value=0.5, step=0.05, description='Cutoff:', style={'description_width': '120px'}, layout={'width': '400px'})
        # For calibrating the images
        self.calibration_checkbox = Checkbox(value=False, description='FFT Calibration')
        self.image_selector = widgets.Dropdown(options=[(f'Image {i}', i) for i in range(len(self.stack.raw_data))], value=0, description='Calibrate Image:')
        self.image_selector.observe(self.update_reference_image, names='value')


        # Initialize reference image from selector
        self.ref_image_index = self.image_selector.value
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]
        self.calibrate_region_checkbox = Checkbox(value=False, description='Calibrate Specific Region')
        self.region_x = IntSlider(min=0, max=4096, value=0, description='X:', layout={'display': 'none'})
        self.region_y = IntSlider(min=0, max=4096, value=0, description='Y:', layout={'display': 'none'})
        self.region_width = IntText(value=100, description='Width:', layout={'display': 'none'})
        self.region_height = IntText(value=100, description='Height:', layout={'display': 'none'})
        self.apply_calibration_checkbox = Checkbox(value=False, description='Apply Calibration to All Images', layout={'width': '400px'})
        self.update_reference_image({'new': self.ref_image_index}) 


        # Save FFT image button
        self.save_fft_button = widgets.Button(description='Save FFT Image')
        self.fft_image_name = Text(value='fft_calibration.svg', description='Image name:', style={'description_width': '100px'}, layout={'width': '400px'})
        self.save_fft_button.on_click(self.save_fft_image)
        self.angle_multilayer_slider = FloatSlider(min=0, max=360, value=0, step=1, description='Angle Multilayer:', style={'description_width': '200px'}, layout={'width': '400px'})
        max_layers = 5  # Based on your calibration_types
        self.layer_angles = [FloatSlider(min=0,max=360,value=0,step=1,description=f'Layer {i} Angle (°)',style={'description_width': '140px'},layout={'width': '300px', 'display': 'none'}) for i in range(2, max_layers+1)]  # Creates sliders for layers 2-5
        self.fft_order_dropdown = Dropdown(options=self.fft_orders, value='1st-order',description='FFT Order:')
        self.save_for_figure_checkbox = Checkbox(value=False, description='Save for Figure', layout={'width': '400px'}) 
        # For plotting and controlling the fft widgets
        calibration_widgets = [ self.image_selector, self.calibration_type_dropdown, self.materials_dropdown, self.min_sampling_slider, self.max_sampling_slider,
                               self.gamma_fft_slider, self.gaussian_sigma_fft_slider,
                                self.region_x, self.region_y, self.region_width, self.region_height, self.calibrate_region_checkbox, self.n_widget_sliders, self.gamma_slider
                                , self.gaussian_sigma_slider, self.contrast_slider, self.double_gaussian_slider1, self.double_gaussian_slider2, self.double_gaussian_weight_slider,
                                self.kernel_size_slider, self.colormap_dropdown, self.fft_spots_rotation_slider, self.rolloff_slider, self.cuttoff_slider, self.angle_multilayer_slider,
                                self.resize_factor, *self.layer_angles, self.save_for_figure_checkbox, self.fft_order_dropdown]
        for widget in calibration_widgets:
            widget.observe(self.fft_calibrate, names='value')
        self.calibrate_button = widgets.Button(description='Calibrate')
        self.calibrate_button.on_click(self.fft_calibrate)
        self.calibration_output = Output()
        self.calibration_display = Output()

        #Clibration_controls
        self.calibration_controls = VBox([
            HBox([self.image_selector, self.materials_dropdown, self.calibration_type_dropdown, self.fft_order_dropdown]),
            HBox([self.region_x, self.region_y, self.region_width, self.region_height, self.calibrate_region_checkbox]),
            HBox([self.n_widget_sliders, self.fft_spots_rotation_slider, self.min_sampling_slider, self.max_sampling_slider]),
            HBox([self.gamma_fft_slider, self.gaussian_sigma_fft_slider, self.cuttoff_slider, self.rolloff_slider]),
            HBox([ self.save_fft_button, self.fft_image_name, self.save_fov_button, self.fov_input, self.save_for_figure_checkbox]),
            HBox([self.calibrate_button, self.apply_calibration_checkbox ]),  
            HBox(self.layer_angles),
            self.calibration_output,
            self.calibration_display
        ], layout={'display': 'none'})
        self.calibration_checkbox.observe(self.handle_calibration_checkbox, names='value')
        self.calibration_type_dropdown.observe(self.update_layer_controls, names='value')
        self.fft_order_dropdown.observe(self.update_fft_order, names='value')

        # self.resize_checkbox.observe(self.resize_image, names='value')



        ############################# Image plotting with a scalebar configuration widgets ###############################
        self.specific_region_checkbox = Checkbox(value=False, description='Save specific region')
        self.scalebar_length_slider = IntSlider(min=1, max=400, value=10, step=1, description='Scalebar length (nm)', style={'description_width': '200px'}, layout={'display': 'none','width': '400px'})
        self.dpi_slider = IntSlider(min=100, max=1200, value=300, step=100, description='DPI', style={'description_width': '160px'}, layout={'display': 'none','width': '400px'})
        self.scalebar_length_checkbox = Checkbox(value=False, description='Add specific scalebar length (nm)')
        self.dpi_checkbox = Checkbox(value=False, description='Save image with DPI')
        self.image_name = Text(value='image.png', description='Image name:', style={'description_width': '100px'}, layout={'width': '400px'})
        self.image_format_dropdown = Dropdown(options=['png', 'svg'], value='png', description='Image format:', style={'description_width': '160px'}, layout={'width': '300px'})
        self.save_image_button = Button(description="Save Image", tooltip="Save image with scalebar", layout={'width': '160px'})
        self.save_for_figure_button = Button(description="Save Figure", tooltip="Save image with scalebar", layout={'width': '160px'})


        #############################  Morphological operations widgets #######################################
        self.opening_slider = IntSlider(min=0, max=10, value=0, description='Opening Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.closing_slider = IntSlider(min=0, max=10, value=0, description='Closing Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.dilation_slider = IntSlider(min=0, max=10, value=0, description='Dilation Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.erosion_slider = IntSlider(min=0, max=10, value=0, description='Erosion Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.gradient_slider = IntSlider(min=0, max=10, value=0, description='Gradient Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.boundary_slider = IntSlider(min=0, max=10, value=0, description='Boundary Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.black_hat_slider = IntSlider(min=0, max=10, value=0, description='Black Hat Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.top_hat_slider = IntSlider(min=0, max=10, value=0, description='Top Hat Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.opening_slider2 = IntSlider(min=0, max=10, value=0, description='Opening Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.closing_slider2 = IntSlider(min=0, max=10, value=0, description='Closing Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.dilation_slider2 = IntSlider(min=0, max=10, value=0, description='Dilation Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.erosion_slider2 = IntSlider(min=0, max=10, value=0, description='Erosion Iterations 2', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.gradient_slider2 = IntSlider(min=0, max=10, value=0, description='Gradient Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        self.boundary_slider2 = IntSlider(min=0, max=10, value=0, description='Boundary Iterations 2', style={'description_width': '160px'}, layout={'display': 'none', 'width': '360px'})
        
        # For calculating clusters and single atom density in clean graphene and contamination
        self.mask_color_dropdown = Dropdown(options=['red', 'white','black', 'green', 'blue'], value='red', description='Mask Color',continuous_update=False)
        self.filename_input = Text(value='analysis_results', description='Filename:', style={'description_width': '100px'}, layout={'width': '400px'})
        self.num_atoms_input = IntText(value=0, description='Num Atoms')
        self.save_button = Button(description="Save", tooltip="Save the data point to a CSV file. A new file will be created if one does not exist", layout={'width': '160px'})
        self.save_button.on_click(self.save_data)

        ####################################  Checkboxes  ######################################
        self.kernel_size_checkbox = Checkbox(value=False, description='Choose Kernel Size')
        self.opening_checkbox = Checkbox(value=False, description='Apply Opening')
        self.closing_checkbox = Checkbox(value=False, description='Apply Closing')
        self.dilation_checkbox = Checkbox(value=False, description='Apply Dilation')
        self.erosion_checkbox = Checkbox(value=False, description='Apply Erosion')
        self.gradient_checkbox = Checkbox(value=False, description='Apply Gradient')
        self.boundary_checkbox = Checkbox(value=False, description='Apply Boundary Extraction')
        self.black_hat_checkbox = Checkbox(value=False, description='Apply Black Hat')
        self.top_hat_checkbox = Checkbox(value=False, description='Apply Top Hat')

        self.min_area_checkbox = Checkbox(value=False, description='Min clean_cont area')
        self.max_area_checkbox = Checkbox(value=False, description='Max clean_cont area')
        self.min_area_checkbox_sa = Checkbox(value=False, description='Min cluster_sa area')
        self.max_area_checkbox_sa = Checkbox(value=False, description='Max cluster_sa area')
        self.circularity_checkbox = Checkbox(value=False, description='Min Circularity')
        self.isolation_checkbox = Checkbox(value=False, description='Min Isolation')

        self.opening_checkbox2 = Checkbox(value=False, description='Apply Opening2')
        self.closing_checkbox2 = Checkbox(value=False, description='Apply Closing2')
        self.dilation_checkbox2 = Checkbox(value=False, description='Apply Dilation2')
        self.erosion_checkbox2 = Checkbox(value=False, description='Apply Erosion2')
        self.gradient_checkbox2 = Checkbox(value=False, description='Apply Gradient2')
        self.boundary_checkbox2 = Checkbox(value=False, description='Apply Boundary Extraction2')
        self.single_atom_clusters_definer_checkbox = Checkbox(value=False, description='Single Atom Clusters Definer')
        self.make_circular_thresh_checkbox = Checkbox(value=False, description='Make circular thresh')

        ########################### Filters Checkboxes ###################################
        self.contrast_checkbox = Checkbox(value=False, description='Contrast Enhancement')
        self.gaussian_checkbox = Checkbox(value=False, description='Gaussian Blur')
        self.double_gaussian_checkbox = Checkbox(value=False, description='Double Gaussian Filter')
        self.brightness_checkbox = Checkbox(value=False, description='Brightness Adjustment')
        self.sigmoid_checkbox = Checkbox(value=False, description='Sigmoid Contrast')
        self.log_transform_checkbox = Checkbox(value=False, description='Log Transform')
        self.exp_transform_checkbox = Checkbox(value=False, description='Exp Transform')
        self.clahe_checkbox = Checkbox(value=False, description='Apply CLAHE')
        self.gamma_checkbox = Checkbox(value=False, description='Gamma Correction')

                
        self.save_image_button.on_click(self.save_image)
        ########################### For observing checkboxes ###################################
        checkboxes = [
            self.kernel_size_checkbox, self.contrast_checkbox, self.gaussian_checkbox,
            self.double_gaussian_checkbox, self.clahe_checkbox, self.gamma_checkbox,
            self.opening_checkbox, self.closing_checkbox, self.dilation_checkbox,
            self.opening_checkbox2, self.closing_checkbox2, self.dilation_checkbox2,
            self.erosion_checkbox2, self.gradient_checkbox2, self.boundary_checkbox2,
            self.erosion_checkbox, self.gradient_checkbox, self.boundary_checkbox,
            self.black_hat_checkbox, self.top_hat_checkbox, self.min_area_checkbox, self.isolation_checkbox, 
            self.max_area_checkbox, self.min_area_checkbox_sa, self.max_area_checkbox_sa, self.circularity_checkbox, 
            self.scalebar_length_checkbox, self.dpi_checkbox, self.single_atom_clusters_definer_checkbox,
            self.brightness_checkbox, self.sigmoid_checkbox, self.log_transform_checkbox,
            self.exp_transform_checkbox, self.make_circular_thresh_checkbox,self.specific_region_checkbox]
        for cb in checkboxes:
            cb.observe(self.toggle_visibility, names='value')


        

        # Layout
        self.controls = VBox([
            HBox([self.contrast_slider, self.gamma_slider, self.clahe_clip_slider, self.clahe_tile_slider]),
            HBox([self.gaussian_sigma_slider, self.double_gaussian_slider1, self.double_gaussian_slider2, self.double_gaussian_weight_slider]),
            HBox([self.brightness_slider, self.sigmoid_alpha, self.sigmoid_beta]),
            HBox([self.contour_retrieval_dropdown, self.contour_approximation_dropdown]),
            HBox([ self.kernel_size_slider, self.min_isolation_slider, self.circularity_slider]),
            HBox([ self.single_atom_clusters_definer, self.make_circular_thresh]),
            HBox([self.min_area_slider, self.max_area_slider, self.min_area_sa_clusters, self.max_area_sa_clusters]),
            HBox([self.opening_slider, self.closing_slider, self.erosion_slider, self.dilation_slider, self.gradient_slider]),
            HBox([self.opening_slider2, self.closing_slider2, self.dilation_slider2, self.erosion_slider2, self.gradient_slider2]),
            HBox([self.save_button, self.filename_input, self.mask_color_dropdown, self.save_for_figure_button, self.image_name]), 
            HBox([self.calibration_checkbox]),
            HBox([self.calibration_controls]),
        ])



        # Layout
        self.controls_image_with_scalebar = VBox([
            HBox([self.contrast_slider, self.gamma_slider, self.clahe_clip_slider, self.clahe_tile_slider]),
            HBox([self.brightness_slider, self.sigmoid_alpha, self.sigmoid_beta, self.opening_slider]),
            HBox([self.gaussian_sigma_slider, self.kernel_size_slider, self.scalebar_length_slider, self.dpi_slider]),
            HBox([self.double_gaussian_slider1, self.double_gaussian_slider2, self.double_gaussian_weight_slider]),
            HBox([self.region_x, self.region_y, self.region_width, self.region_height]),
            HBox([self.save_image_button, self.image_name, self.image_format_dropdown, self.resize_factor, self.resize_method]),
            HBox([self.line_profile_widgets, self.region_profile_widgets]),
            HBox([self.calibration_checkbox]),
            HBox([self.calibration_controls]),
        ])



        if self.analysing_features:
            display(VBox([  
                HBox([self.contrast_checkbox, self.gaussian_checkbox, self.double_gaussian_checkbox, self.clahe_checkbox, self.gamma_checkbox ]),
                HBox([self.brightness_checkbox, self.sigmoid_checkbox, self.log_transform_checkbox, self.exp_transform_checkbox, self.kernel_size_checkbox,]),
                HBox([self.opening_checkbox, self.closing_checkbox, self.dilation_checkbox, self.erosion_checkbox, self.gradient_checkbox]),
                HBox([self.opening_checkbox2, self.closing_checkbox2, self.dilation_checkbox2, self.erosion_checkbox2, self.gradient_checkbox2]),
                HBox([self.min_area_checkbox, self.max_area_checkbox, self.min_area_checkbox_sa, self.max_area_checkbox_sa]),
                HBox([self.single_atom_clusters_definer_checkbox, self.circularity_checkbox, self.isolation_checkbox,  self.make_circular_thresh_checkbox]),
                HBox([self.slice_slider, self.threshold_slider, self.threshold_slider_sa, self.colormap_dropdown]),
                self.controls,
                self._interactive_image_analysis()
            ]))
        elif self.save_images_with_calibrated_scalebar:
            display(VBox([  
                HBox([self.contrast_checkbox, self.gaussian_checkbox, self.double_gaussian_checkbox, self.clahe_checkbox, self.gamma_checkbox, self.opening_checkbox]),
                HBox([self.brightness_checkbox, self.sigmoid_checkbox, self.log_transform_checkbox, self.exp_transform_checkbox, self.kernel_size_checkbox]),
                HBox([self.slice_slider, self.colormap_dropdown, self.specific_region_checkbox, self.resize_checkbox]),
                self.controls_image_with_scalebar,

                
                self._interactive_image_analysis()
            ]))


    def define_global_font_matplotlib(self):
        from matplotlib import font_manager
            # Register the font
        font_manager.fontManager.addfont(self.font_path)
        # Extract font name
        import os
        font_name = font_manager.FontProperties(fname=self.font_path).get_name()
        # Set globally
        plt.rcParams['font.family'] = font_name


    def resize_image(self, image):
        """Enhanced resizing with dynamic factor and method selection"""
        if self.resize_checkbox.value and self.resize_factor.value != 1.0:
            image = image.copy()  # Create a copy to avoid modifying the original
            method = cv2.INTER_CUBIC if self.resize_method.value == 'Bicubic' else cv2.INTER_LINEAR
            new_size = (int(image.shape[1] * self.resize_factor.value),
                        int(image.shape[0] * self.resize_factor.value))
            return cv2.resize(image, new_size, interpolation=method)
        return image



    def update_layer_controls(self, change):
            """Update visible layer angle sliders based on selected layers"""
            calib_type = self.calibration_type_dropdown.value
            
            if calib_type.startswith('graphene-'):
                try:
                    layers = int(calib_type.split('-')[1][:-1])  # Extract number from 'graphene-XL'
                    visible_sliders = layers - 1  # For N layers, need N-1 angles
                except (IndexError, ValueError):
                    visible_sliders = 0
            else:
                visible_sliders = 0

            # Update slider visibility
            for i, slider in enumerate(self.layer_angles):
                slider.layout.display = 'flex' if i < visible_sliders else 'none'

    def update_fft_order(self, change):
        fft_order = self.fft_order_dropdown.value

                

    def update_reference_image(self, change):
        """Update reference image when selector changes"""
        self.ref_image_index = change['new']
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]
        # print(f"Updated reference image to index {self.ref_image_index}")
        self._get_ref_fov()



    def update_colormap(self, change):
        """Handles colormap changes without triggering a full update."""
        self.update(colormap=self.colormap_dropdown.value)
    
    def update_mask_color(self, change):
        """Handles mask color changes without triggering a full update."""
        self.update(mask_color=self.mask_color_dropdown.value)


    def toggle_visibility(self, change):
        # Map checkboxes to their corresponding sliders
        checkbox_slider_map = {
            self.kernel_size_checkbox: [self.kernel_size_slider],
            self.contrast_checkbox: [self.contrast_slider],
            self.gaussian_checkbox: [self.gaussian_sigma_slider],
            self.double_gaussian_checkbox: [self.double_gaussian_slider1, 
                                        self.double_gaussian_slider2,
                                        self.double_gaussian_weight_slider],
            self.clahe_checkbox: [self.clahe_clip_slider, self.clahe_tile_slider],
            self.gamma_checkbox: [self.gamma_slider],
            self.brightness_checkbox: [self.brightness_slider],
            self.sigmoid_checkbox: [self.sigmoid_alpha, self.sigmoid_beta],
            self.log_transform_checkbox: [],
            self.exp_transform_checkbox: [],
            self.opening_checkbox: [self.opening_slider],
            self.closing_checkbox: [self.closing_slider],
            self.dilation_checkbox: [self.dilation_slider],
            self.erosion_checkbox: [self.erosion_slider],
            self.gradient_checkbox: [self.gradient_slider],
            self.boundary_checkbox: [self.boundary_slider],
            self.black_hat_checkbox: [self.black_hat_slider],
            self.top_hat_checkbox: [self.top_hat_slider],
            self.min_area_checkbox: [self.min_area_slider],
            self.max_area_checkbox: [self.max_area_slider],
            self.min_area_checkbox_sa: [self.min_area_sa_clusters],
            self.max_area_checkbox_sa: [self.max_area_sa_clusters],
            self.make_circular_thresh_checkbox: [self.make_circular_thresh],
            self.circularity_checkbox: [self.circularity_slider], 
            self.isolation_checkbox: [self.min_isolation_slider],
            self.opening_checkbox2: [self.opening_slider2],
            self.closing_checkbox2: [self.closing_slider2],
            self.dilation_checkbox2: [self.dilation_slider2],
            self.erosion_checkbox2: [self.erosion_slider2],
            self.gradient_checkbox2: [self.gradient_slider2],
            self.boundary_checkbox2: [self.boundary_slider2],
            self.scalebar_length_checkbox: [self.scalebar_length_slider],
            self.dpi_checkbox: [self.dpi_slider],
            self.single_atom_clusters_definer_checkbox: [self.single_atom_clusters_definer],
            self.specific_region_checkbox: [self.region_x, self.region_y, self.region_width, self.region_height],
            self.resize_checkbox: [self.resize_factor, self.resize_method],
            }

        # Find which checkbox changed and update its corresponding sliders
        for checkbox, sliders in checkbox_slider_map.items():
            if change['owner'] == checkbox:
                visibility = '' if change['new'] else 'none'
                for slider in sliders:
                    slider.layout.display = visibility
                break


    def handle_calibration_checkbox(self, change):
        """Handles the main calibration checkbox state"""
        if change['new']:
            # Show controls and run initial calibration
            self.calibration_controls.layout.display = ''
            self.fft_calibrate()
        else:
            # Hide controls and clear outputs
            self.calibration_controls.layout.display = 'none'
            with self.calibration_display:
                clear_output(wait=True)
            with self.calibration_output:
                clear_output(wait=True)




######################################### FFT calibration related methods #########################################
    def fft_calibrate(self, b=None):
        """Updates calibration display when parameters change"""
        if not self.calibration_checkbox.value:
            return
        
        try:

            # Get reference image and parameters
            ref_image = self.ref_image.copy()
            gamma = self.gamma_slider.value
            gaussian_sigma = self.gaussian_sigma_slider.value
            contrast = self.contrast_slider.value
            double_gaussian_sigma1 = self.double_gaussian_slider1.value
            double_gaussian_sigma2 = self.double_gaussian_slider2.value
            double_gaussian_weight = self.double_gaussian_weight_slider.value
            kernel = self.kernel_size_slider.value
            colormap = self.colormap_dropdown.value
            fft_order = int(self.fft_order_dropdown.value[0])
            print(f"FFT Order: {fft_order}")
            ref_image = self.apply_filters_fft(ref_image, gamma, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel)
            ref_image_plotiing = ref_image.copy()
            # Apply region selection if enabled
            if self.calibrate_region_checkbox.value:
                x = self.region_x.value
                y = self.region_y.value
                w = self.region_width.value
                h = self.region_height.value
                ref_image = ref_image[y:y+h, x:x+w]
            layer_angles=[slider.value for slider in self.layer_angles if slider.layout.display != 'none']
            # Create calibrator with current parameters
            self.calibrator = fsc.FourierSpaceCalibrator(
                template=self.calibration_type_dropdown.value,
                lattice_constant=self.materials[self.materials_dropdown.value],
                max_sampling=self.max_sampling_slider.value,
                min_sampling=self.min_sampling_slider.value,
                normalize_azimuthal=False,
                layer_angles=layer_angles,
                fft_order=fft_order,)
            
            # Window FFT parameters
            rolloff = self.rolloff_slider.value
            cutoff = self.cuttoff_slider.value

            # Perform calibration
            self.calibration_factor = self.calibrator.calibrate(ref_image) / 10

            # Update display
            with self.calibration_display:
                clear_output(wait=True)
                if not hasattr(self, 'cal_fig') or not plt.fignum_exists(self.cal_fig.number):
                    self.cal_fig, self.cal_ax1 = plt.subplots(1, 2, figsize=(16, 8))
                    self.cal_fig.show()
                else:
                    # Clear previous content
                    for ax in self.cal_ax1.flatten():
                        ax.cla()
                    
                # Show selected region
                self.current_ref_image_fft = ref_image
                self.cal_ax1[0].imshow(ref_image_plotiing, cmap=colormap)
                if self.calibrate_region_checkbox.value:
                    rect = patches.Rectangle((x, y), w, h, 
                                        linewidth=2, edgecolor='red', facecolor='none')
                    self.cal_ax1[0].add_patch(rect)
                self.cal_ax1[0].set_title(f'Calibration Region (Image {self.ref_image_index})')
                gamma_fft = self.gamma_fft_slider.value
                gaussian_sigma_fft = self.gaussian_sigma_fft_slider.value
                # Show FFT analysis
                ft_image = np.fft.fftshift(fsc.windowed_fft(ref_image, cf=cutoff, rf=rolloff))
                ft_vis = np.log(np.abs(ft_image) + 1e-6)
                ft_vis = self.apply_filters_fft_spots(ft_vis, gamma_fft, gaussian_sigma_fft)
                spots = self.calibrator.get_spots()
                spots = fsc.rotate(spots, self.fft_spots_rotation_slider.value, center=(ft_vis.shape[0] // 2, ft_vis.shape[1] // 2))
                n = self.n_widget_sliders.value
                # Current FFT data for saving
                self.current_spots = spots
                self.current_n = self.n_widget_sliders.value
                self.current_colormap = colormap
                self.current_ft_vis = ft_vis
                self.cal_ax1[1].imshow(ft_vis, cmap=colormap)
                if self.current_n <= 150:
                    marker_size = 7 * (160/self.current_n)
                elif 150<self.current_n < 300:
                    marker_size = 12 * (160/self.current_n)
                else:
                    marker_size = 20 * (160/self.current_n)
                self.cal_ax1[1].plot(*spots.T, 'wo', mfc='none', markersize= marker_size, alpha=1)
                self.cal_ax1[1].set_xlim(ft_vis.shape[0] // 2 - n, ft_vis.shape[0] // 2 + n)
                self.cal_ax1[1].set_ylim(ft_vis.shape[1] // 2 - n, ft_vis.shape[1] // 2 + n)
                self.cal_ax1[1].set_title('Detected Spots')
                
                # Redraw only the changed elements
                for ax in self.cal_ax1.flatten():
                    ax.axis('off')
                    
                self.cal_fig.tight_layout()
                self.cal_fig.canvas.draw()

            # Update calibration results
            with self.calibration_output:
                clear_output(wait=True)
                print(f'Calibration factor: {self.calibration_factor:.6f} nm/pixel')
                print(f'Reference FOV: {self.ref_fov} nm')
                print(f'Calibrated FOV: {self.ref_image_shape * self.calibration_factor:.3f} nm')
                print('Apply calibration using checkbox below')

        except Exception as e:
            with self.calibration_output:
                print(f'Calibration failed: {str(e)}')



    def save_fft_image(self, b):

        """Saves both the calibration image and FFT analysis as separate SVG files"""
        if not hasattr(self, 'current_ft_vis') or not hasattr(self, 'current_ref_image_fft'):
            with self.calibration_output:
                print("No images to save. Please run calibration first")
            return

        base_name = self.fft_image_name.value.replace('.svg', '')
        ref_filename = f"{base_name}_slice({self.ref_image_index})_ref.svg"
        fft_filename = f"{base_name}_slice({self.ref_image_index})_fft.svg"
        if '/' in ref_filename:
            ref_img_fft = ref_filename.split('/')[-1]
            dir_name  = ref_filename.split('/')[0]
            os.makedirs(dir_name, exist_ok=True)
            ref_filename = os.path.join(dir_name, ref_img_fft)
        if '/' in fft_filename:
            fft_img_fft = fft_filename.split('/')[-1]
            dir_name  = fft_filename.split('/')[0]
            os.makedirs(dir_name, exist_ok=True)
            fft_filename = os.path.join(dir_name, fft_img_fft)


        fig_ref, ax_ref = plt.subplots(figsize=(16, 8), dpi=300)
        ax_ref.imshow(self.current_ref_image_fft, cmap=self.current_colormap)

        # Draw region rectangle if region selection is active
        if self.calibrate_region_checkbox.value:
            x = self.region_x.value
            y = self.region_y.value
            w = self.region_width.value
            h = self.region_height.value
            rect = patches.Rectangle((x, y), w, h, 
                                linewidth=1, edgecolor='red', facecolor='none')
            ax_ref.add_patch(rect)

        # Add informational box
        if not self.save_for_figure_checkbox.value:
            info_text = (f"Slice: {self.ref_image_index}\ncalibration factor: {self.calibration_factor:.6f} nm/pixel\n"
                        f"FOV: {self.ref_fov:.2f} nm\nCalibrated FOV: {self.ref_image_shape * self.calibration_factor:.2f} nm")
            anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=12),
                                        frameon=True, pad=0.5, borderpad=0.5)
            anchored_text.patch.set_boxstyle("round,pad=0.3")
            anchored_text.patch.set_facecolor("white")
            anchored_text.patch.set_alpha(0.9)
            ax_ref.add_artist(anchored_text)

            ax_ref.axis('off')
            plt.savefig(ref_filename, bbox_inches='tight', format='svg', pad_inches=0)
            plt.close(fig_ref)

            # Save FFT analysis
            fig_fft, ax_fft = plt.subplots(figsize=(5, 5), dpi=200)
            ax_fft.imshow(self.current_ft_vis, cmap=self.current_colormap)
            ax_fft.plot(*self.current_spots.T, 'wo', mfc='none', 
                    markersize=5 * (140/self.current_n), alpha=0.5)
            ax_fft.set_xlim(self.current_ft_vis.shape[0]//2 - self.current_n,
                        self.current_ft_vis.shape[0]//2 + self.current_n)
            ax_fft.set_ylim(self.current_ft_vis.shape[1]//2 - self.current_n,
                        self.current_ft_vis.shape[1]//2 + self.current_n)
            

            # Extract visible angles
            visible_angles = [slider.value for slider in self.layer_angles if slider.layout.display != 'none']
            angle_text = ", ".join([f"{angle:.2f}°" for angle in visible_angles])

            # Compose info box content
            txt_angle_mismatches = f"Angle mismatches: {angle_text}" if visible_angles and self.calibration_type_dropdown.value not in ['hexagonal', '2nd-order-hexagonal'] else " "
            info_text = (
                f"FFT calibration of the image (slice {self.ref_image_index})\n"
                f"Calibration factor: {self.calibration_factor:.5f} nm/pixel\n"
                f"{txt_angle_mismatches}")

            anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=12),
                                        frameon=True, pad=0.5, borderpad=0.5)
            anchored_text.patch.set_boxstyle("round,pad=0.3")
            anchored_text.patch.set_facecolor("white")
            anchored_text.patch.set_alpha(0.9)
            ax_fft.add_artist(anchored_text)


            ax_fft.axis('off')
            plt.savefig(fft_filename, bbox_inches='tight', format='svg', pad_inches=0)
            plt.close(fig_fft)

        else:

            # Extract visible angles
            visible_angles = [slider.value for slider in self.layer_angles if slider.layout.display != 'none']
            angle_text = ", ".join([f"{angle:.2f}°" for angle in visible_angles])
            txt_angle_mismatches = f"Angle mismatches: {angle_text}" if visible_angles and self.calibration_type_dropdown.value not in ['hexagonal', '2nd-order-hexagonal'] else " "
            # Compose info box content
            if self.calibration_type_dropdown.value not in ['hexagonal', '2nd-order-hexagonal']:
                info_text = (f"Number of layers: {self.calibration_type_dropdown.value[-2]}\n"
                    f"{txt_angle_mismatches}")
                # Save the images with the same name
                fig_ref, ax_ref = plt.subplots(figsize=(16, 8), dpi=300)
                anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=12),
                                frameon=True, pad=0.5, borderpad=0.5)
                anchored_text.patch.set_boxstyle("round,pad=0.3")
                anchored_text.patch.set_facecolor("white")
                anchored_text.patch.set_alpha(0.9)
                ax_ref.add_artist(anchored_text)

            else:
                fig_ref, ax_ref = plt.subplots(figsize=(16, 8), dpi=300)
            ax_ref.imshow(self.current_ref_image_fft, cmap=self.current_colormap)
            ax_ref.axis('off')
            plt.savefig(ref_filename, bbox_inches='tight', format='svg', pad_inches=0)
            plt.close(fig_ref)

            # Save FFT analysis
            fig_fft, ax_fft = plt.subplots(figsize=(5, 5), dpi=200)
            ax_fft.imshow(self.current_ft_vis, cmap=self.current_colormap)

            if self.calibration_type_dropdown.value not in ['hexagonal', '2nd-order-hexagonal']:
                anchored_text1 = AnchoredText(info_text, loc='upper left', prop=dict(size=12),
                    frameon=True, pad=0.5, borderpad=0.5)
                ax_fft.add_artist(anchored_text1)

            if self.current_n <= 150:
                marker_size = 7 * (160/self.current_n)
            elif 150<self.current_n < 300:
                marker_size = 12 * (160/self.current_n)
            else:
                marker_size = 20 * (160/self.current_n)
            ax_fft.plot(*self.current_spots.T, 'wo', mfc='none', 
                    markersize= marker_size, alpha=0.5)
            ax_fft.set_xlim(self.current_ft_vis.shape[0]//2 - self.current_n,
                        self.current_ft_vis.shape[0]//2 + self.current_n)
            ax_fft.set_ylim(self.current_ft_vis.shape[1]//2 - self.current_n,
                        self.current_ft_vis.shape[1]//2 + self.current_n)


            ax_fft.axis('off')
            plt.savefig(fft_filename, bbox_inches='tight', format='svg', pad_inches=0)
            plt.close(fig_fft)

        with self.calibration_output:
            print(f"Saved:\n- Calibration image: {ref_filename}\n- FFT analysis: {fft_filename}")


    def display_calibration_region(self, image_index, x, y, w, h):
        """Displays the selected calibration region on the original image"""
        with self.calibration_display:
            clear_output(wait=True)
            image = self.stack.raw_data[image_index].copy()
            image = improve_contrast(image)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image, cmap='gray')
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'Calibration Region (Image {image_index})')
            plt.show()




    def get_calibrated_image(self, image, slice_number):

        if self._last_calibration == (slice_number, self.calibration_factor):
            return self._cached_nm_per_pixel, self._cached_nm_per_pixel**2
        # Calibration
        meta = self.metadata[f"metadata_{slice_number:04d}"]
        fov = self.metadata.get_specific_metadata('fov_nm', required_keys=['scan_device_properties'], data=meta)
        if len(fov) > 0:
            fov = fov[0]
        else:
            fov = self.metadata.get_specific_metadata('scale', required_keys=['spatial_calibrations'], data=meta)
            if len(fov) > 0:
                fov = fov[0] * image.shape[0]
            else:
                fov = 1
                print(f'There is no FOV in the metadata. Using a diffualt FOV of 1 nm')
            
        if (self.calibration_factor and hasattr(self, 'ref_fov') and self.apply_calibration_checkbox.value and self.ref_fov is not None):
            # print(meta)
            # print(f'FOV from metadata: {fov} nm')
            # print(f'Image shape: {image.shape[0]} pixels')
            # print(f'Reference image shape: {self.ref_image_shape} pixels')
            scale_factor = (self.ref_image_shape / image.shape[0]) * (fov / self.ref_fov)
            nm_per_pixel = self.calibration_factor * scale_factor
            print(scale_factor, nm_per_pixel)
            nm2_per_pixel2 = nm_per_pixel ** 2
            fov_calibrated = image.shape[0] * nm_per_pixel
            print("Original image shape", image.shape[0],  "Reference image shape", self.ref_image_shape)
            print(f'Calibrated FOV: {fov_calibrated:.2f} nm', 'FOV from metadata:', fov)

        else:
            print('Not calibrated yet, using a default values of 1 nm/pixel')
            nm_per_pixel = 1
            nm2_per_pixel2 = 1

        self._last_calibration = (slice_number, self.calibration_factor)
        self._cached_nm_per_pixel = nm_per_pixel
        return nm_per_pixel, nm2_per_pixel2


    def get_VGVOACurrent(self):
        """Get the VG-VOA current from the metadata"""
        if self.metadata is not None:
            metadata = self.metadata[f"metadata_{self.ref_image_index:04d}"]
            vgvoa_current = self.metadata.get_specific_metadata('BP2_^VGVOACurrent', required_keys=['instrument'], data=metadata)
            if len(vgvoa_current) > 0:
                vgvoa_current = vgvoa_current[0]
            else:
                vgvoa_current = 1
                print('No VG-VOA current found in the metadata. Careful, using a default value of 1 ms.')
        else:
            print('No metadata available for this reference image. Please use another image for the FFT calibration.')
        return vgvoa_current
    

    def get_dwell_time(self):
        """Get the dwell time from the metadata"""
        if self.metadata is not None:
            metadata = self.metadata[f"metadata_{self.ref_image_index:04d}"]
            dwell_time = self.metadata.get_specific_metadata('pixel_time_us', required_keys=['scan_device_properties'], data=metadata)
            if len(dwell_time) > 0:
                dwell_time = dwell_time[0]
            else:
                dwell_time = 1
                print('No dwell time found in the metadata. Careful, using a default value of 1 ms.')
        else:
            print('No metadata available for this reference image. Please use another image for the FFT calibration.')
        return dwell_time
  


    def _get_ref_fov(self):
        if self.metadata is not None:
            metadata = self.metadata[f"metadata_{self.ref_image_index:04d}"]
            self.ref_fov = self.metadata.get_specific_metadata('fov_nm', required_keys=['scan_device_properties'], data=metadata)
            if len(self.ref_fov) > 0:
                self.ref_fov = self.ref_fov[0]
            else:
                self.ref_fov = self.metadata.get_specific_metadata('scale', required_keys=['spatial_calibrations'], data=metadata)
                if len(self.ref_fov) > 0:
                    self.ref_fov = self.ref_fov[0] * self.ref_image_shape
                else:
                    self.ref_fov = 1
                    print('No FOV found in the metadata. Careful, using a difual value of 1 nm.')
            # print(f'There is metadata and the reference FOV: {self.ref_fov} nm')
        else:
            print('No metadata available for this reference image. Please use another image for the FFT calibration.')
        return self.ref_fov
            
            

    def save_fov_to_metadata(self, b):
        """Save modified FOV only to scan_device_parameters. I added this because sometimes the FOV is not saved correctly in the scan_device_parameters"""
        try:
            new_fov = self.fov_input.value
            with h5py.File(self.stack_path, 'r+') as hf:
                meta_path = f"metadata/metadata_{self.ref_image_index:04d}"
                meta_json = hf[meta_path][()]
                meta = json.loads(meta_json)
                
                # Update ONLY scan_device_parameters.fov_nm
                if 'metadata' in meta:
                    scan = meta['metadata']['scan']
                    scan['fov_nm'] = new_fov
                    if 'scan_device_parameters' in scan:
                        # Update fov_nm
                        scan['scan_device_parameters']['fov_nm'] = new_fov
                                            # Update related parameters
                        if 'scan_size' in scan:
                            scan_size = scan['scan_size']
                            if scan_size and len(scan_size) >= 1:
                                # Calculate pixel size from FOV and scan dimensions
                                pixel_size = new_fov / max(scan_size)
                                scan['scan_device_parameters']['pixel_size'] = [pixel_size, pixel_size]
                                if 'spatial_calibrations' in meta:
                                    for cal in meta['spatial_calibrations']:
                                        cal['scale'] = pixel_size
                                        cal['units'] = 'nm'
                        
                # Save back to HDF5
                del hf[meta_path]
                hf[meta_path] = np.string_(json.dumps(meta))

            # Refresh metadata
            self.stack = DataLoader(self.stack_path)
            self.metadata = self.stack.raw_metadata
            self._get_ref_fov()

            with self.calibration_output:
                clear_output(wait=True)
                print(f"✅ Updated scan_device_parameters.fov_nm to {new_fov} nm")
                print("Validated changes:")
                print(f"scan_device_parameters.fov_nm: {meta['metadata']['scan']['scan_device_parameters']['fov_nm']}")
        except Exception as e:
            with self.calibration_output:
                print(f"❌ Error saving FOV: {str(e)}")
                import traceback
                traceback.print_exc()



    ################################## Filters functions #####################################

    def improve_contrast(self, image,percentile=10):
        low = np.percentile(image, percentile)
        high = np.percentile(image,100 - percentile)
        return exposure.rescale_intensity(image, in_range=(low, high))
    

    def apply_clahe(self, image, clip_limit, tile_grid_size):
        """Ensures the image is 8-bit before applying CLAHE."""
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(image)


    def apply_gamma_correction(self, image, gamma):
        """Applies gamma correction."""
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)


    def apply_sigmoid_contrast(self, image, alpha, beta):
        normalized = image.astype(np.float32) / 255
        sigmoid = 1 / (1 + np.exp(-alpha * (normalized - beta)))
        return (255 * sigmoid).astype(np.uint8)

    def apply_log_transform(self, image):
        c = 255 / np.log(1 + np.max(image))
        return (c * np.log1p(image)).astype(np.uint8)

    def apply_exp_transform(self, image, gamma):
        return (255 * (image/255) ** gamma).astype(np.uint8)

    def apply_unsharp_mask(self, image, sigma, strength):
        blurred = cv2.GaussianBlur(image, (0,0), sigma)
        return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

    def apply_retinex(self, image, sigma):
        log_image = cv2.log(image.astype(np.float32) + 1)
        blur = cv2.GaussianBlur(log_image, (0,0), sigma)
        retinex = cv2.exp(log_image - blur)
        return cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)




    # Modified morphological operations to use dynamic kernel size
    def dilate(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(image, kernel, iterations=iterations)

    def erode(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.erode(image, kernel, iterations=iterations)

    def opening(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)

    def closing(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    def gradient(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)

    def boundary_extraction(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(image, kernel, iterations=iterations)
        return cv2.subtract(image, eroded)

    def black_hat(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

    def top_hat(self, image, iterations, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)


    # # Morphological operations
    def apply_morphological_operations(self, thresh, iteration_opening, iteration_closing, iteration_dilation, iteration_erosion, 
                                       iteration_gradient, iteration_boundary, iteration_black_hat, iteration_top_hat, kernel):
        if self.erosion_checkbox.value:
            thresh = self.erode(thresh, iteration_erosion, kernel)
        if self.dilation_checkbox.value:
            thresh = self.dilate(thresh, iteration_dilation, kernel)
        if self.opening_checkbox.value:
            thresh = self.opening(thresh, iteration_opening, kernel)
        if self.closing_checkbox.value:
            thresh = self.closing(thresh, iteration_closing, kernel)
        if self.gradient_checkbox.value:
            thresh= self.gradient(thresh, iteration_gradient, kernel)
        if self.boundary_checkbox.value:
            thresh = self.boundary_extraction(thresh, iteration_boundary, kernel)
        if self.black_hat_checkbox.value:
            thresh = self.black_hat(thresh, iteration_black_hat, kernel)
        if self.top_hat_checkbox.value:
            thresh = self.top_hat(thresh, iteration_top_hat, kernel)
        return thresh
    


    def apply_morphological_operations2(self, image, opening2, closing2, dilation2, erosion2, gradient2, boundary2, kernel):
        if self.erosion_checkbox2.value:
            image = self.erode(image, erosion2, kernel)
        if self.dilation_checkbox2.value:
            image = self.dilate(image, dilation2, kernel)
        if self.opening_checkbox2.value:
            image = self.opening(image, opening2, kernel)
        if self.closing_checkbox2.value:
            image = self.closing(image, closing2, kernel)
        if self.gradient_checkbox2.value:
            image = self.gradient(image, gradient2, kernel)
        if self.boundary_checkbox2.value:
            image = self.boundary_extraction(image, boundary2, kernel)
        return image
    


        # Apply different filters
    def apply_filters_fft(self, image, gamma, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel):
        if self.gaussian_checkbox.value:
            kernel_tuple = (kernel, kernel)  # Converting integer kernel size to tuple
            image = cv2.GaussianBlur(image, kernel_tuple, gaussian_sigma)

        # Apply contrast enhancement
        if self.contrast_checkbox.value:
            image = improve_contrast(image, contrast)  #Global contrast correction across the image

        # Apply Double Gaussian Filter
        if self.double_gaussian_checkbox.value:
            image = double_gaussian(image, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight)

        #Apply gammaa correction
        if self.gamma_checkbox.value:
            image = self.apply_gamma_correction(image, gamma)
        return image
    


    def apply_filters_fft_spots(self, image, gamma, gaussian_sigma):
        from scipy.ndimage import gaussian_filter
        if self.gaussian_checkbox.value:
            image = gaussian_filter(image, sigma=gaussian_sigma)
        if self.gamma_checkbox.value:
            image = self.apply_gamma_correction(image, gamma)
        return image
    

    # Apply different filters
    def apply_filters(self, image, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast,
                 double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight,
                 kernel, brightness, sigmoid_alpha, sigmoid_beta):
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if self.gaussian_checkbox.value:
            kernel_tuple = (kernel, kernel)  # Converting integer kernel size to tuple
            image = cv2.GaussianBlur(image, kernel_tuple, gaussian_sigma)


        # Apply contrast enhancement
        if self.contrast_checkbox.value:
            image = improve_contrast(image, contrast)  #Global contrast correction across the image

        # Apply Double Gaussian Filter
        if self.double_gaussian_checkbox.value:
            image = double_gaussian(image, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight)


        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gamma correction (for brightness adjustment)
         #Local contrast correction across each tile of the image
        if self.clahe_checkbox.value:
            image = self.apply_clahe(image, clahe_clip, clahe_tile)

        if self.gamma_checkbox.value:
            image = self.apply_gamma_correction(image, gamma)

        # New filters
        if self.brightness_checkbox.value:
            image = cv2.add(image, brightness)

        if self.sigmoid_checkbox.value:
            image = self.apply_sigmoid_contrast(image, sigmoid_alpha, sigmoid_beta)

        if self.log_transform_checkbox.value:
            image = self.apply_log_transform(image)

        if self.exp_transform_checkbox.value:
            image = self.apply_exp_transform(image, gamma)

        return image



    def add_line_profile_widgets(self):
        """Create widgets for line profile parameters"""
        self.line_x = IntSlider(min=0, max=1650, value=0, description='Line X:')
        self.line_y = IntSlider(min=0, max=1650, value=0, description='Line Y:')
        self.line_length = IntSlider(min=1, max=500, value=0, description='Length:')
        self.line_width = IntSlider(min=1, max=50, value=0, description='Width:')
        self.line_angle = FloatSlider(min=0.2, max=360, value=0, description='Angle:')
        self.save_line_profile_button = Button(description="Save Line Profile")
        self.save_line_profile_button.on_click(self.save_line_profile)
        self.line_profile_name = Text(value='line_profile', description='File name:')
        
        return VBox([
            HBox([self.line_x, self.line_y]),
            HBox([self.line_length, self.line_width, self.line_angle]),
            HBox([self.save_line_profile_button, self.line_profile_name]),
        ])

    def add_region_profile_widgets(self):
        """Create widgets for region profile parameters"""
        self.region_profile_x = IntSlider(min=0, max=1650, value=0, description='Region X:')
        self.region_profile_y = IntSlider(min=0, max=1650, value=0, description='Region Y:')
        self.region_profile_width = IntSlider(min=1, max=500, value=0, description='Width:')
        self.region_profile_height = IntSlider(min=1, max=500, value=0, description='Height:')
        self.save_region_profile_button = Button(description="Save Region Data")
        self.save_region_profile_button.on_click(self.save_region_profile)
        self.region_profile_name = Text(value='region_profile', description='File name:')
        
        return VBox([
            HBox([self.region_profile_x, self.region_profile_y]),
            HBox([self.region_profile_width, self.region_profile_height]),
            HBox([self.save_region_profile_button, self.region_profile_name]),
        ])
    

    def get_line_profile(self, image, x, y, length, width, angle):
        """Calculate intensity profile along a line"""
        from skimage.measure import profile_line
        
        # Convert to image coordinates (row, column)
        start = (y, x)  # (row, column)
        
        # Calculate end point in image coordinates
        angle_rad = np.deg2rad(angle)
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)
        end = (y + dy, x + dx)  # (row, column)
        
        # Get profile with specified width
        profile = profile_line(image, start, end, 
                            linewidth=width, 
                            reduce_func=np.mean)
        # Considering dwell time and VG-VOA current
        dwell_time = self.get_dwell_time()
        vgvoa_current = self.get_VGVOACurrent()
        profile = profile / ( dwell_time * vgvoa_current)
        return profile


    def get_region_profile(self, image, x, y, w, h):
        """Calculate average intensity profile in a region"""
        region = image[y:y+h, x:x+w]
        # Considering dwell time and VG-VOA current
        dwell_time = self.get_dwell_time()
        vgvoa_current = self.get_VGVOACurrent()
        print(dwell_time, vgvoa_current)
        region = region / (dwell_time * vgvoa_current)
        print('region', region.mean(), region.shape, type(region))
        return region.mean()  # we can also specifiy alonge which axis to get the profile using np.mean(axis=0) or np.mean(axis=1)

    def save_line_profile(self, b):
        """Save line profile plot and data"""
        params = {
            'image': self.stack.raw_data[self.slice_slider.value],
            'x': self.line_x.value,
            'y': self.line_y.value,
            'length': self.line_length.value,
            'width': self.line_width.value,
            'angle': self.line_angle.value
        }
        
        profile = self.get_line_profile(**params)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(profile)
        ax.set_title(f"Line Intensity Profile\n{params['length']}px length, {params['width']}px width")
        ax.set_xlabel("Position along line")
        ax.set_ylabel("Intensity")
        plt.savefig(f"line_profile_{timestamp}.png")
        plt.close()
        line_profile_name = self.line_profile_name.value
        # Save data
        np.savetxt(f"{line_profile_name}.csv", profile, delimiter=",")

    

    def save_region_profile(self, b):
        """Save region profile data and marked image"""
        params = {
            'image': self.stack.raw_data[self.slice_slider.value],
            'x': self.region_profile_x.value,
            'y': self.region_profile_y.value,
            'w': self.region_profile_width.value,
            'h': self.region_profile_height.value
        }
        
        profile = self.get_region_profile(**params)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save marked image
        fig, ax = plt.subplots()
        ax.imshow(params['image'], cmap='gray')
        rect = plt.Rectangle((params['x'], params['y']), params['w'], params['h'],
                            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.savefig(f"region_marked_{timestamp}.png")
        plt.close()

        region_profile_name = f"{self.region_profile_name.value}"

        # Save data
        np.savetxt(f"{region_profile_name}_w{self.region_profile_width.value}_h{self.region_profile_height.value}.csv", [profile], delimiter=",")




    def display_calibrated_image_with_scalebar(self, gamma, clahe_clip, clahe_tile, gaussian_sigma,contrast, double_gaussian_sigma1, double_gaussian_sigma2,
                                         double_gaussian_weight, kernel, brightness, sigmoid_alpha,sigmoid_beta, scalebar_length, exp_transform, log_transform, 
                                         x,y,h,w, resize_factor,resize_method, opening, line_x, line_y, line_length, line_width, line_angle,
                                        region_profile_x, region_profile_y, region_profile_width, region_profile_height, colormap=None, slice_number=None):
        
        
        with self.calibration_display:
            clear_output(wait=True)
        
        # Process image
        image = self.stack.raw_data[slice_number]
        print("Minimum value:", np.min(image), "Maximum value:", np.max(image))
        nm_per_pixel, _ = self.get_calibrated_image(image, slice_number)

        image = self.closing(image, opening, kernel)

        print(f"Image {slice_number} shape:", image.shape)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255 
        image = self.apply_filters(image, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel,
                      brightness, sigmoid_alpha, sigmoid_beta)
        image_1 = image.copy()
        if self.resize_checkbox.value:
            interpolation_method = self.INTERPOLATION_MAP[self.resize_method.value]
            image = cv2.resize(image, (int(image.shape[1] * self.resize_factor.value), int(image.shape[0] * self.resize_factor.value)), interpolation=interpolation_method)

        # get histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        bins_center = (bins[:-1] + bins[1:]) / 2

        
        
        # Create/maintain figure
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            self.fig, self.axs = plt.subplots(2, 2, figsize=(24, 8), dpi=100)
            self.fig.subplots_adjust(0, 0, 1, 1)
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False

        # Clear previous content
        for ax in self.axs.flatten():
            ax.cla()
            ax.axis('off')



        self.axs[0,0].imshow(image, cmap=colormap)
        self.add_scalebar(self.axs[0,0], nm_per_pixel, scalebar_length)
        self.axs[0,0].set_title('Original Image')
                
        if self.specific_region_checkbox.value is True:
            x = self.region_x.value
            y = self.region_y.value
            w = self.region_width.value
            h = self.region_height.value
            rect = patches.Rectangle((x, y), w, h,
                                    linewidth=1, edgecolor='red', facecolor='none')
            self.axs[0,0].add_patch(rect)
            image_1 = image[y:y+h, x:x+w]
            self.axs[1,0].imshow(image_1, cmap=colormap)

        else:
            self.axs[1,0].imshow(image_1, cmap=colormap)
            self.add_scalebar(self.axs[1,0], nm_per_pixel, scalebar_length)
        # Add intensity profile elements
        line_profile = None
        region_profile = None
        
        # Draw line profile if parameters are set
        if any([self.region_x.value > 0, self.region_y.value > 0]):
            # Get values in image coordinates
            length = self.region_width.value
            width = self.region_height.value
            angle = self.line_angle.value

            # Calculate line endpoints
            angle_rad = np.deg2rad(angle)
            dx = length * np.cos(angle_rad)
            dy = length * np.sin(angle_rad)

            # Create polygon for width visualization
            perp_angle = angle_rad + np.pi/2
            dx_perp = width/2 * np.cos(perp_angle)
            dy_perp = width/2 * np.sin(perp_angle)

            x0 = self.region_x.value 
            y0 = self.region_y.value + width/2 * np.sin(perp_angle)

            
            # Calculate polygon vertices in image coordinates
            points = [
                (x0 - dx_perp, y0 - dy_perp),
                (x0 + dx_perp, y0 + dy_perp),
                (x0 + dx + dx_perp, y0 + dy + dy_perp),
                (x0 + dx - dx_perp, y0 + dy - dy_perp)
            ]
            
            # Draw sampling area
            self.axs[0,0].add_patch(plt.Polygon(points, closed=True, 
                                        edgecolor='red', 
                                        facecolor='red', 
                                        alpha=0.3))
            
            # Calculate actual profile
            line_profile = self.get_line_profile(image, x0, y0, length, width, angle)
            
        # Draw region if parameters are set
        if any([self.region_x.value > 0, self.region_y.value > 0]):
            rx = self.region_x.value
            ry = self.region_y.value
            rw = self.region_width.value
            rh = self.region_height.value
            
            # Draw rectangle on image
            rect = patches.Rectangle((rx, ry), rw, rh, 
                                linewidth=2, edgecolor='cyan', facecolor='none')
            self.axs[0,0].add_patch(rect)
            
            # Calculate region profile
            region_profile = self.get_region_profile(image, rx, ry, rw, rh)

        self.axs[0,0].set_title('Cropped region')
        v_min, v_max = image.min(), image.max()
        hist_max = hist.max()

        if v_max != v_min:
            # Create mask for current display range
            mask = (bins_center >= v_min) & (bins_center <= v_max)
            
        # Histogram plot
        self.axs[0,1].plot(bins_center, hist, color='gray', alpha=0.7)
        self.axs[0,1].fill_between(bins_center, hist, color='gray', alpha=0.4)  # Corrected line
        self.axs[0,1].set_title('Histogram')
        self.axs[0,1].set_xlabel('Pixel Intensity')
        self.axs[0,1].set_ylabel('Frequency')
        self.axs[0,1].set_xlim(0, 255)
        self.axs[0,1].set_ylim(0, np.max(hist) * 1.1)
        self.axs[0,1].axis('on')
        self.axs[0,1].set_facecolor('#f0f0f0')
        self.axs[0,1].grid(True, linestyle='--', alpha=0.7)

        # Intensity Profiles
        if line_profile is not None or region_profile is not None:
            self.axs[1,1].axis('on')  # Force axis to be visible
            self.axs[1,1].set_facecolor('#f0f0f0')  
            self.axs[1,1].grid(True, linestyle='--', alpha=0.7)
            
            # Set spine properties
            for spine in self.axs[1,1].spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#404040')
                spine.set_linewidth(0.8)

            # Customize ticks and labels
            self.axs[1,1].tick_params(axis='both', which='major', 
                                labelsize=10, color="#155DBC")
            self.axs[1,1].set_xlabel('Position', fontsize=12, color='#303030')
            self.axs[1,1].set_ylabel('Intensity', fontsize=12, color='#303030')
            self.axs[1,1].set_title('Intensity Profiles', fontsize=14, pad=15)

            # Plot styling
            if line_profile is not None:
                self.axs[1,1].plot(line_profile, 
                                color="#344ab8", 
                                linewidth=2,
                                linestyle='-',
                                marker='o',
                                markersize=4,
                                label='Line Profile')
            if region_profile is not None:
                self.axs[1,1].plot(region_profile, 
                                color='#0066cc', 
                                linewidth=2,
                                linestyle='--',
                                marker='s',
                                markersize=4,
                                label='Region Profile')
                
            # Add legend with nicer styling
            legend = self.axs[1,1].legend(frameon=True, 
                                    fontsize=10,
                                    facecolor='white',
                                    edgecolor='#404040',
                                    loc='upper right')
            legend.get_frame().set_linewidth(0.8)

        else:
            self.axs[1,1].axis('off')

        # Adjust layout with padding
        self.fig.tight_layout(pad=4.0)
        self.fig.subplots_adjust(wspace=0.15, hspace=0.15)
        self.fig.canvas.draw_idle()
        return image, image_1



    def save_image(self, _):
        # Get current parameters
        params = {
            'gamma': self.gamma_slider.value,
            'clahe_clip': self.clahe_clip_slider.value,
            'clahe_tile': self.clahe_tile_slider.value,
            'gaussian_sigma': self.gaussian_sigma_slider.value,
            'contrast': self.contrast_slider.value,
            'double_gaussian_sigma1': self.double_gaussian_slider1.value,
            'double_gaussian_sigma2': self.double_gaussian_slider2.value,
            'double_gaussian_weight': self.double_gaussian_weight_slider.value,
            'scalebar_length': self.scalebar_length_slider.value,
            'colormap': self.colormap_dropdown.value,
            'slice_number': self.slice_slider.value,
            'kernel': self.kernel_size_slider.value,
            'brightness': self.brightness_slider.value,
            'sigmoid_alpha': self.sigmoid_alpha.value,
            'sigmoid_beta': self.sigmoid_beta.value,
            'exp_transform': self.exp_transform_checkbox.value,
            'log_transform': self.log_transform_checkbox.value,
            'contrast': self.contrast_slider.value,
            'x' : self.region_x.value,
            'y' : self.region_y.value,
            'h' : self.region_height.value,
            'w' : self.region_width.value,  
            'resize_factor': self.resize_factor.value,
            'resize_method': self.resize_method.value,  
            'opening': self.opening_slider.value,
            'line_x': self.line_x.value,
            'line_y': self.line_y.value,
            'line_length': self.line_length.value,
            'line_width': self.line_width.value,
            'line_angle': self.line_angle.value,
            'region_profile_x': self.region_profile_x.value,
            'region_profile_y': self.region_profile_y.value,
            'region_profile_width': self.region_profile_width.value,
            'region_profile_height': self.region_profile_height.value,      
        }
        
        # Create dedicated figure for vector output
        dpi = self.dpi_slider.value
        fig = plt.figure(figsize=(8, 8), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])

        # Generate and display image
        if self.specific_region_checkbox.value is True:
            image = self.display_calibrated_image_with_scalebar(**params)[1]
        else:
            image = self.display_calibrated_image_with_scalebar(**params)[0]
        nm_per_pixel, _ = self.get_calibrated_image(image, self.slice_slider.value)
        nm_per_pixel = nm_per_pixel / params['resize_factor']
        ax.imshow(image, cmap=params['colormap'])
        self.add_scalebar(ax, nm_per_pixel, params['scalebar_length'])
        rect1 = plt.Rectangle((params['region_profile_x'], params['region_profile_y']), params['region_profile_width'], params['region_profile_height'],
                                linewidth=1, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect1)

        # Add rotated line profile polygon
        x0 = params['line_x']
        y0 = params['line_y']
        length = params['line_length']
        width = params['line_width']
        angle = params['line_angle']

        angle_rad = np.deg2rad(angle)
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)

        perp_angle = angle_rad + np.pi/2
        dx_perp = width/2 * np.cos(perp_angle)
        dy_perp = width/2 * np.sin(perp_angle)

        points = [
            (x0 - dx_perp, y0 - dy_perp),
            (x0 + dx_perp, y0 + dy_perp),
            (x0 + dx + dx_perp, y0 + dy + dy_perp),
            (x0 + dx - dx_perp, y0 + dy - dy_perp)
        ]

        polygon = plt.Polygon(points, closed=True, 
                            edgecolor='yellow', 
                            facecolor='yellow', 
                            alpha=0.7)
        ax.add_patch(polygon)
        ax.axis('off')
        file_format = self.image_format_dropdown.value.lower()  # e.g., "png" or "svg"

        filename = f"{self.image_name.value}.{file_format}"
        if '/' in filename:
            image_name = filename.split('/')[-1]
            dir_name = filename.split('/')[0]
        else:
            image_name = filename
            dir_name = '.'

        os.makedirs(dir_name, exist_ok=True)
        file_path = os.path.join(dir_name, image_name)

        fig.savefig(file_path, format=file_format, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Image saved: {file_path}")



    def add_scalebar(self, ax, nm_per_pixel, scalebar_length, colormap=None, float_scale=False):
        """Add vector scale bar directly to existing axes"""
        # Clear previous scale bar elements
        for artist in ax.artists:
            if isinstance(artist, (plt.Rectangle, plt.Text)):
                artist.remove()
        
        # Get current image dimensions
        img_size = ax.images[0].get_array().shape[0]
        shift_pixels = img_size // 16
        
        # Calculate scale bar parameters
        if self.fixed_position_scalebar:
            scale_bar_pixels = img_size // 4
            scale_bar_length_nm = scale_bar_pixels * nm_per_pixel
        else:
            scale_bar_pixels = int(scalebar_length / nm_per_pixel)

        if scale_bar_length_nm > 10:
            scale_bar_length_nm = round(scale_bar_length_nm / 4) * 4
        elif 10>=scale_bar_length_nm >= 1.5:
            scale_bar_length_nm = round(scale_bar_length_nm/2) * 2
        elif 0.5 < scale_bar_length_nm < 1.5:
            scale_bar_length_nm = 1
        elif 0.3 <scale_bar_length_nm <= 0.5:
            scale_bar_length_nm = 0.4
        elif 0 <scale_bar_length_nm <= 0.3:
            scale_bar_length_nm = 0.2

        scale_bar_pixels = int(scale_bar_length_nm / nm_per_pixel)  # Converting back to pixels
        # Convert to axis coordinates (bottom-left origin)
        ax_start_x = shift_pixels / img_size
        ax_start_y = (shift_pixels * 0.66) / img_size
        scale_bar_length_frac = scale_bar_pixels / img_size
        scale_bar_thickness = max(0.01, 1.6/img_size)

        # Create vector scale bar
        scale_bar = plt.Rectangle(
            (ax_start_x, ax_start_y), 
            scale_bar_length_frac, 
            scale_bar_thickness,
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='black',
            linewidth=0.3,
            zorder=8
        )
        ax.add_artist(scale_bar)

        # Add text with original sizing logic
        if scale_bar_length_nm >=1:
            text = f"{scale_bar_length_nm:.0f} nm"
        else:
            text = f"{scale_bar_length_nm:.1f} nm"
        text_x = ax_start_x + scale_bar_length_frac/2
        text_y = ax_start_y + scale_bar_thickness * 2
        base_image_size_ref_font = 512
        # Font size calculation matching original image.size/16 ratio
        font_size = (img_size / 16) * (base_image_size_ref_font/img_size)  # Converting to points
        
        text_artist = ax.text(
            text_x, text_y, text,
            transform=ax.transAxes,
            color='white',
            ha='center',
            va='bottom',
            fontsize=font_size,
            fontproperties=FontProperties(fname=self.font_path),
            path_effects=[patheffects.withStroke(linewidth=1, foreground="black")]
        )
        return ax


    def mask_color(self, mask_color):
        if mask_color == 'red':
            return [255, 0, 0]
        elif mask_color == 'white':
            return [255, 255, 255]
        elif mask_color == 'black':
            return [0, 0, 0]
        elif mask_color == 'green':
            return [0, 255, 0]
        elif mask_color == 'blue':
            return [0, 0, 255]
        else:
            return [0, 0, 0]



    def measure_circularity(self, contour):
        """Calculate circularity of a contour (4π*Area/Perimeter²)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter ** 2)

    def measure_roundness(self, contour):
        """Calculate roundness (minor axis/major axis) using fitted ellipse"""
        if len(contour) < 5:  # Need at least 5 points to fit ellipse
            return 0.0
        (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
        if major_axis == 0:
            return 0.0
        return minor_axis / major_axis



    def measure_aspect_ratio(self, contour):
        """Calculate aspect ratio (width/height) using bounding rectangle"""
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return 0.0
        return float(w) / h
    
    

    def measure_feret_diameter(self, contour, nm_px):
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



    def contour_min_distance(self, cnt1, cnt2):
        """Optimized distance calculation with adaptive point sampling"""
        if cnt1 is cnt2:
            return float('inf')
        
        min_dist = float('inf')
        
        # Determine sampling rate based on contour complexity
        def get_step(contour):
            length = len(contour)
            if length <= 20:    # Check all points for small contours
                return 1
            elif length <= 50:  # Check every 3rd point
                return 3
            else:               # Check every 5th point for large contours
                return 5
        
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

    def calculate_isolation(self, contours, nm_per_pixel, isolation_distance):
        """Optimized isolation check with spatial partitioning"""
        n = len(contours)
        isolation_mask = np.ones(n, dtype=bool)
        
        if n < 2:
            return isolation_mask
        
        isolation_px = max(isolation_distance / nm_per_pixel, 0.1)
        
        # Create spatial index using bounding circles
        circles = [cv2.minEnclosingCircle(c) for c in contours]
        centers = np.array([(x, y) for (x, y), _ in circles])
        radii = np.array([r for _, r in circles])
        
        # Find potential neighbors using spatial partitioning
        tree = KDTree(centers)
        pairs = tree.query_pairs(np.max(radii) * 2 + isolation_px * 2)
        
        # Check only potentially close pairs
        for i, j in pairs:
            contour_i_area = cv2.contourArea(contours[i]) * nm_per_pixel**2
            contour_j_area = cv2.contourArea(contours[j]) * nm_per_pixel**2
            if contour_i_area <0.025 or contour_j_area < 0.025:
                continue
            if (np.linalg.norm(centers[i] - centers[j]) - (radii[i] + radii[j])) > isolation_px:
                continue
            
            dist = self.contour_min_distance(contours[i], contours[j])
            if dist <= isolation_px:
                isolation_mask[i] = False
                isolation_mask[j] = False
        
        return isolation_mask



    def update(self, threshold, threshold_sa, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast, min_clean_cont_area, min_cluster_area,
            max_cluster_area, max_clean_cont_area, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, 
            iteration_opening, iteration_closing, iteration_dilation, iteration_erosion, iteration_gradient, min_circularity, isolation_distance,
            iteration_boundary, iteration_black_hat, iteration_top_hat, opening2, closing2, dilation2, erosion2, gradient2, boundary2, sa_cluster_definer,
            brightness, sigmoid_alpha, sigmoid_beta, exp_transform, log_transform, make_circular,
            contour_retrieval_modes, contour_approximation_methods, colormap=None, slice_number=None, kernel=None, masked_color=None, display_images=True):
        """
        Full function with circularity filtering and isolated atom detection
        isolation_distance: Minimum pixel distance between atoms to consider them isolated (default 15)
        """
        
        with self.calibration_display:
            clear_output(wait=True)

        # Load and preprocess image
        image = self.stack.raw_data[slice_number]
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255 
        image = self.apply_filters(image, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel,
                      brightness, sigmoid_alpha, sigmoid_beta)
        image = np.uint8(image)

        # Thresholding and morphology
        thresh_type = cv2.THRESH_BINARY_INV if self.clean_graphene_analysis else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(image, threshold, 255, thresh_type)
        thresh = self.apply_morphological_operations(thresh, iteration_opening, iteration_closing, iteration_dilation,
                                                    iteration_erosion, iteration_gradient, iteration_boundary,
                                                    iteration_black_hat, iteration_top_hat, kernel)

        nm_per_pixel, nm2_per_pixel2 = self.get_calibrated_image(image, slice_number)


        # Clean area detection
        contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_contours = [cnt for cnt in contours if min_clean_cont_area <= (cv2.contourArea(cnt) * nm2_per_pixel2) < max_clean_cont_area]
        contour_mask = np.zeros_like(image)
        cv2.drawContours(contour_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

        # Cluster/atom detection with isolation filtering
        masked_clean_image = cv2.bitwise_and(image, image, mask=contour_mask)
        if self.clusters_sa_analysis:
            _, thresh_sa = cv2.threshold(masked_clean_image, threshold_sa, 255, cv2.THRESH_BINARY)
        elif self.defects_analysis:
            _, thresh_sa = cv2.threshold(masked_clean_image, threshold_sa, 255, cv2.THRESH_BINARY_INV)
        
        else:
            pass
        thresh_sa = self.apply_morphological_operations2(thresh_sa, opening2, closing2, dilation2, erosion2, gradient2, boundary2, kernel)


        # # Find clusters and single atoms
        # contours_sa, _ = cv2.findContours(thresh_sa, contour_retrieval_modes, contour_approximation_methods)
        # Get integer values from dropdown selections
        retrieval_mode = self.contour_retrieval_modes[contour_retrieval_modes]
        approx_method = self.contour_approximation_methods[contour_approximation_methods]

        contours_sa, _ = cv2.findContours(thresh_sa, retrieval_mode, approx_method)

        # Convert to integer coordinates
        contours_sa = [c.astype(np.int32) for c in contours_sa]

        # Calculate isolation status for ALL original SA contours
        if self.isolation_checkbox.value:
            isolation_mask = self.calculate_isolation(contours_sa, nm_per_pixel, isolation_distance)
        else:
            isolation_mask = np.ones(len(contours_sa), dtype=bool)

        # Independent condition checks
        valid_contours_sa = []
        centroids = []
        for idx, cnt in enumerate(contours_sa):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Independent area check
            area_nm2 = area * nm2_per_pixel2
            area_ok = (min_cluster_area <= area_nm2 < max_cluster_area)
            
            # Independent circularity check
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            circ_ok = circularity >= min_circularity
            
            # Independent isolation check
            isol_ok = isolation_mask[idx]
            
            # Combine independent conditions
            if area_ok and circ_ok and isol_ok:
                valid_contours_sa.append(cnt)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append([cx, cy])


        circularity =  [self.measure_circularity(cnt) for cnt in valid_contours_sa]
        roundness = [self.measure_roundness(cnt) for cnt in valid_contours_sa]
        feret_diameter = [self.measure_feret_diameter(cnt,nm_per_pixel) for cnt in valid_contours_sa]
        aspect_ratio = [self.measure_aspect_ratio(cnt) for cnt in valid_contours_sa]
        
        # Prepare data for histograms
        clean_areas = [cv2.contourArea(cnt) * nm2_per_pixel2 for cnt in valid_contours]
        cluster_areas = [cv2.contourArea(cnt) * nm2_per_pixel2 for cnt in valid_contours_sa]


       #Results calculation
        if self.clean_graphene_analysis:
            total_clean_area_nm2 = sum(cv2.contourArea(cnt) * nm2_per_pixel2 for cnt in valid_contours)
            clusters_in_clean_graphene = [cv2.contourArea(cnt) * nm2_per_pixel2 for cnt in valid_contours_sa]
            single_atoms = [element for element in clusters_in_clean_graphene if element < sa_cluster_definer]
            clusters = [element for element in clusters_in_clean_graphene if element >= sa_cluster_definer]
            number_of_atoms = len(single_atoms)
            number_of_clusters = len(clusters)
            total_sa_cluster_area_nm2 = sum(clusters_in_clean_graphene)
            total_atoms_area = sum(single_atoms)
            total_clusters_area = sum(clusters)
            density_atoms_in_clean_graphene = number_of_atoms / total_clean_area_nm2 if total_clean_area_nm2 > 0 else float('nan')
            density_clusters_in_clean_graphene = number_of_clusters / total_clean_area_nm2 if total_clean_area_nm2 > 0 else float('nan')

        elif self.contamination_analysis:
            total_contamination_area_nm2 = sum(cv2.contourArea(cnt) * nm2_per_pixel2 for cnt in valid_contours)  
            clusters_in_contamination = [cv2.contourArea(cnt) * nm2_per_pixel2 for cnt in valid_contours_sa] 
            single_atoms = [element for element in clusters_in_contamination if element <= sa_cluster_definer]
            clusters = [element for element in clusters_in_contamination if element > sa_cluster_definer]
            number_of_atoms = len(single_atoms)
            number_of_clusters = len(clusters)
            total_sa_cluster_area_nm2 = sum(clusters_in_contamination)
            total_atoms_area = sum(single_atoms)
            total_clusters_area = sum(clusters)
            density_atoms_in_contamination = number_of_atoms / total_contamination_area_nm2 if total_contamination_area_nm2 > 0 else float('nan')
            density_clusters_in_contamination = number_of_clusters / total_contamination_area_nm2   if total_contamination_area_nm2 > 0 else float('nan')

        results = {
                        'clean_graphene': {},
                        'contamination': {},
                        'clusters_and_single_atoms': {}
                    }

        if self.clean_graphene_analysis:
            results['clean_graphene'].update({'number_of_clusters': number_of_clusters,                
                'total_area_nm2': total_clean_area_nm2,
                'total_cluster_area_nm2': total_clusters_area,
                'clusters_density_in_graphene_nm2': density_clusters_in_clean_graphene,
                'clusters': clusters,
                'number_of_atoms': number_of_atoms,
                'total_atoms_area_nm2': total_atoms_area,
                'atoms_density_nm2': density_atoms_in_clean_graphene,
                'atoms': single_atoms,
                'shape_metrics': {
                    'circularity': circularity,
                    'roundness': roundness,
                    'feret_diameter': feret_diameter,
                    'aspect_ratio': aspect_ratio
                }}
            )
            results['clusters_and_single_atoms']['total_cluster_area_nm2'] = total_sa_cluster_area_nm2

        elif self.contamination_analysis:
            results['contamination'].update({
                'total_area_nm2': total_contamination_area_nm2,
                'Clusters_analysis': {
                    'number_of_clusters': number_of_clusters,
                    'total_area_nm2': total_clusters_area,
                    'clusters_density_in_contamination_nm2': density_clusters_in_contamination,
                    'clusters': clusters
                },
                'Atoms_analysis': {
                    'number_of_atoms': number_of_atoms,
                    'total_atoms_area_nm2': total_atoms_area,
                    'atoms_density_in_contamination_nm2': density_atoms_in_contamination,
                    'atoms': single_atoms
                },
                'shape_metrics': {
                    'circularity': circularity,
                    'roundness': roundness,
                    'feret_diameter': feret_diameter,
                    'aspect_ratio': aspect_ratio
                }
            })
            results['clusters_and_single_atoms']['total_cluster_area_nm2'] = total_sa_cluster_area_nm2

            results['clusters_and_single_atoms'].update({
                'detected_clusters_count': len(valid_contours_sa),
                'calibration_nm_per_pixel': nm_per_pixel,
                'contour_data': {
                    'clean_areas_nm2': clean_areas if 'clean_areas' in locals() else [],
                    'cluster_areas_nm2': cluster_areas if 'cluster_areas' in locals() else []
                }
            })
        if self.clean_graphene_analysis or self.contamination_analysis:
            results_key = 'clean_graphene' if self.clean_graphene_analysis else 'contamination'
            results[results_key]['shape_metrics'] = {
                'circularity': circularity,
                'roundness': roundness,
                'feret_diameter': feret_diameter,
                'aspect_ratio': aspect_ratio
            }

        # print(  results['clean_graphene'].get('total_area_nm2', float('nan')),
        #         # results['clean_graphene'].get('number_of_clusters', float('nan')),
        #         # results['clean_graphene'].get('total_cluster_area_nm2', float('nan')),
        #         # results['clean_graphene'].get('clusters_density_in_graphene_nm2', float('nan')),
        #         # results['clean_graphene'].get('clusters', float('nan')),
        #         results['clean_graphene'].get('number_of_atoms', np.nan),
        #         results['clean_graphene'].get('total_atoms_area_nm2', np.nan),
        #         results['clean_graphene'].get('atoms_density_nm2', np.nan),
        #         results['clean_graphene'].get('atoms', np.nan))


        print(  'Total area nm2: ' + f"{results['contamination'].get('total_area_nm2', float('nan'))}" + '\n',
                # results['contamination'].get('Clusters_analysis', {}).get('count', float('nan')),
                # results['contamination'].get('Clusters_analysis', {}).get('total_area_nm2', float('nan')),
                # results['contamination'].get('Clusters_analysis', {}).get('clusters_density_in_contamination_nm2', float('nan')),
                # results['contamination'].get('Clusters_analysis', {}).get('clusters', float('nan')),
                'Number of atoms: '+ f"{results['contamination'].get('Atoms_analysis', {}).get('number_of_atoms', np.nan)}" + '\n',
                # results['contamination'].get('Atoms_analysis', {}).get('total_atoms_area_nm2', np.nan),
                'Atom density: '+ f"{results['contamination'].get('Atoms_analysis', {}).get('atoms_density_in_contamination_nm2', np.nan)}" + '\n',
                results['contamination'].get('Atoms_analysis', {}).get('atoms', np.nan))
        # Display images if requested 
        area_of_interest_label = "Clean graphene area" if self.clean_graphene_analysis else "Contamination area"
        area_of_interest = total_clean_area_nm2 if self.clean_graphene_analysis else total_contamination_area_nm2


        # Visualization

        # Replace small contours with circles of the same area (This is for plotting purposes, it doesn't affect the analysis)
        small_area_threshold_nm2 = make_circular  # Example: 0.5 nm²

        modified_contours = []
        modified_centroids = []  # To store centroids of modified contours
        for cnt in valid_contours_sa:
            area_px = cv2.contourArea(cnt)
            area_nm2 = area_px * nm2_per_pixel2
            
            if area_nm2 < small_area_threshold_nm2:
                # Compute centroid of the original contour
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Generate a circle with the same area as the original contour
                radius = np.sqrt(area_px / np.pi)
                circle_contour = np.array([
                    [   [int(round(cx + radius * np.cos(theta))),
                        int(round(cy + radius * np.sin(theta)))]
                        for theta in np.linspace(0, 2*np.pi, 36) ]], dtype=np.int32)
                modified_contours.append(circle_contour)
                modified_centroids.append([cx, cy])  # Centroid remains the same
            else:
                modified_contours.append(cnt)
                # Recompute centroid for consistency (optional)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    modified_centroids.append([cx, cy])
                else:
                    modified_centroids.append([0, 0])



        masked_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        masked_image[contour_mask == 0] = self.mask_color(masked_color)
        cv2.drawContours(masked_image, valid_contours_sa, -1, (0, 255, 0), 1)


 
        if display_images:
            scalebar_length = self.scalebar_length_slider.value
            valid_contour_sa_mask = np.zeros_like(image)
            cv2.drawContours(valid_contour_sa_mask, modified_contours, -1, 255, thickness=cv2.FILLED)

            # Create or reuse figure
            if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
                self.fig, self.axs = plt.subplots(2, 4, figsize=(16, 8))
                self.fig.show()
            else:
                # Clear previous content
                for ax in self.axs.flatten():
                    ax.cla()

            # Update plots with new data
            self.axs[0,0].imshow(image, cmap=colormap)
            self.axs[0,0].set_title('Original Image')
            self.add_scalebar(self.axs[0,0], nm_per_pixel, scalebar_length)
            
            self.axs[0,1].imshow(thresh, cmap='gray')
            self.axs[0,1].set_title('Thresholding of original image')

            self.axs[0,2].imshow(contour_mask, cmap=colormap)
            self.axs[0,2].set_title('Filtered thresholded image')

            self.axs[0,3].hist(clean_areas, bins=50, color='blue')
            self.axs[0,3].set_xlabel('Clean Area (nm²)')
            self.axs[0,3].set_ylabel('Count')
            self.axs[0,3].set_title('Clean Area Distribution')
            self.axs[0,3].grid(True)

            info_text = (
                f"Number of {area_of_interest_label} found:   {len(clean_areas)}\n"
                f"Total {area_of_interest_label}: {area_of_interest:.2f} nm²\n")

            anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=14),
                                        frameon=True, pad=0.5, borderpad=0.5)
            anchored_text.patch.set_boxstyle("round,pad=0.3")
            anchored_text.patch.set_facecolor("white")
            anchored_text.patch.set_alpha(0.9)
            self.axs[0,3].add_artist(anchored_text)
            
            self.axs[1,0].imshow(masked_image, cmap='gray')
            self.axs[1,0].set_title('Contamination Mask keeping only clean graphene')

            self.axs[1,1].imshow(thresh_sa, cmap=colormap)
            self.axs[1,1].set_title('Threholding of the clean graphene area')
            
            self.axs[1,2].imshow(valid_contour_sa_mask, cmap='gray')
            self.axs[1,2].set_title('Filtered clusters and single atoms')

            self.axs[1,3].hist(cluster_areas, bins=50, color='green')
            self.axs[1,3].set_title('Cluster Size Distribution')
            self.axs[1,3].set_xlabel('Cluster Area (nm²)')
            self.axs[1,3].set_ylabel('Count')
            self.axs[1,3].grid(True)


            info_text = (
                f"Number of clusters found:  {len(cluster_areas)}\n")
                # f"Total cluster area in {area_of_interest_label}: {total_sa_cluster_area_nm2:.2f} nm²\n")
            anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=14),
                                        frameon=True, pad=0.5, borderpad=0.5)
            anchored_text.patch.set_boxstyle("round,pad=0.3")
            anchored_text.patch.set_facecolor("white")
            anchored_text.patch.set_alpha(0.9)
            self.axs[1,3].add_artist(anchored_text)
            # Redraw only the changed elements
            for ax in self.axs.flatten():
                ax.axis('off')
                
            self.fig.tight_layout()
            self.fig.canvas.draw()
            fig_name = f"{self.image_name.value}.svg"

            def save_fig():
                self.fig.savefig(fig_name, format='svg', bbox_inches='tight', pad_inches=0)
                return
        self.save_for_figure_button.on_click(lambda b: save_fig())

        print("Final results:", results)
        self.results = pd.DataFrame(results)
        return results



    def save_new_data_to_existing_df(self, _):
        """Saves/updates ONLY Calibrated_FOV and Entire_Area_nm2 in CSV
        I needed to create this function to save new statistics that I didn't collect yet and 
         I need to add them without overwriting the whole CSV file. This is based on the slice number
         of the image so it will create the new columns and save the new data in the same row as the image slice number."""
        filename = self.filename_input.value
        slice_number = self.slice_slider.value
        
        try:
            # 1. Get calibration data for current slice
            image = self.stack.raw_data[slice_number]
            nm_per_pixel, _ = self.get_calibrated_image(image, slice_number)
            calibrated_fov = image.shape[0] * nm_per_pixel
            entire_area = calibrated_fov ** 2
            
            # 2. Prepare minimal data
            new_data = {
                'Slice': slice_number,
                'Calibrated_FOV': calibrated_fov,
                'Entire_Area_nm2': entire_area
            }
            
            # 3. Read existing CSV or create empty DataFrame
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                # Preserve existing columns
                if 'Calibrated_FOV' not in df.columns:
                    df['Calibrated_FOV'] = np.nan
                if 'Entire_Area_nm2' not in df.columns:
                    df['Entire_Area_nm2'] = np.nan
            else:
                df = pd.DataFrame(columns=['Slice', 'Calibrated_FOV', 'Entire_Area_nm2'])
            
            # 4. Update existing row or append new
            mask = df['Slice'] == slice_number
            if mask.any():
                # Update existing
                df.loc[mask, 'Calibrated_FOV'] = calibrated_fov
                df.loc[mask, 'Entire_Area_nm2'] = entire_area
            else:
                # Append new row (only with these columns)
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df, new_row], ignore_index=True)
            
            # 5. Save back
            df.to_csv(filename, index=False)
            print(f"Updated slice {slice_number}: FOV={calibrated_fov:.2f} nm")
            
        except Exception as e:
            print(f"Save failed: {str(e)}")
            import traceback
            traceback.print_exc()



    def save_data(self, _):
        import json    
        base_name = self.filename_input.value
        if not base_name.endswith('.csv'):
            filename = f"{base_name}.csv"
        else:
            filename = base_name        
        slice_number = self.slice_slider.value
        if '/' in filename:
            fname = filename.split('/')[-1]
            dirname = filename.split('/')[0]
            os.makedirs(dirname, exist_ok=True)
            filename = os.path.join(dirname, fname)

        studied_area = "Clean_Area_nm2" if self.clean_graphene_analysis else "Contamination_Area_nm2"

        # 2. Collect raw parameters directly from widgets
        print("Collecting parameters...")
        params = {
            "threshold": self.threshold_slider.value,
            "threshold_sa": self.threshold_slider_sa.value,
            "gamma": self.gamma_slider.value,
            "clahe_clip": self.clahe_clip_slider.value,
            "clahe_tile": self.clahe_tile_slider.value,
            "gaussian_sigma": self.gaussian_sigma_slider.value,
            "contrast": self.contrast_slider.value,
            "min_clean_cont_area": self.min_area_slider.value,
            "max_clean_cont_area": self.max_area_slider.value,
            "min_cluster_area": self.min_area_sa_clusters.value,
            "max_cluster_area": self.max_area_sa_clusters.value,
            "make_circular": self.make_circular_thresh.value,
            "double_gaussian_sigma1": self.double_gaussian_slider1.value,
            "double_gaussian_sigma2": self.double_gaussian_slider2.value,
            "double_gaussian_weight": self.double_gaussian_weight_slider.value,
            "iteration_opening": self.opening_slider.value,
            "iteration_closing": self.closing_slider.value,
            "iteration_dilation": self.dilation_slider.value,
            "iteration_erosion": self.erosion_slider.value,
            "iteration_gradient": self.gradient_slider.value,
            "min_circularity": self.circularity_slider.value,
            "isolation_distance": self.min_isolation_slider.value,
            "iteration_boundary": self.boundary_slider.value,
            "iteration_black_hat": self.black_hat_slider.value,
            "iteration_top_hat": self.top_hat_slider.value,
            "opening2": self.opening_slider2.value,
            "closing2": self.closing_slider2.value,
            "dilation2": self.dilation_slider2.value,
            "erosion2": self.erosion_slider2.value,
            "gradient2": self.gradient_slider2.value,
            "boundary2": self.boundary_slider2.value,
            "sa_cluster_definer": self.single_atom_clusters_definer.value,
            "slice_number": slice_number,
            "kernel": self.kernel_size_slider.value,
            "brightness": self.brightness_slider.value,
            "sigmoid_alpha": self.sigmoid_alpha.value,
            "sigmoid_beta": self.sigmoid_beta.value,
            "exp_transform": self.exp_transform_checkbox.value,
            "log_transform": self.log_transform_checkbox.value,
            'contour_retrieval_modes': self.contour_retrieval_dropdown.value,
            'contour_approximation_methods': self.contour_approximation_dropdown.value,
            "display_images": False
            
        }

        # 3. Prepare data storage with default NaN values
        new_data = {
            "Slice": slice_number,
            studied_area : np.nan,
            "Number_of_Clusters": np.nan,
            "Total_Cluster_Area_nm2": np.nan,
            "Clusters_Density": np.nan,
            "Clusters": [],
            "Num_Atoms": np.nan,
            "Atoms_Area": np.nan,
            "Atoms_Density": np.nan,
            "Atoms": []
        }

        try:
            results = self.update(**params)
            results_key = 'clean_graphene' if self.clean_graphene_analysis else 'contamination'

            image = self.stack.raw_data[slice_number]
            nm_per_pixel, nm2_per_pixel2 = self.get_calibrated_image(image, slice_number)
            current_fov = image.shape[0] * nm_per_pixel  
            Entire_Area_nm2 = current_fov ** 2
            # 5. Handle results based on analysis type
            if self.clean_graphene_analysis:
                new_data.update({
                    "Clean_Area_nm2": results['clean_graphene'].get('total_area_nm2', np.nan),
                    "Number_of_Clusters": results['clean_graphene'].get('number_of_clusters', np.nan),
                    "Total_Cluster_Area_nm2": results['clean_graphene'].get('total_cluster_area_nm2', np.nan),
                    "Clusters_Density": results['clean_graphene'].get('clusters_density_in_graphene_nm2', np.nan),
                    "Clusters": results['clean_graphene'].get('clusters', []),
                    "Num_Atoms": results['clean_graphene'].get('number_of_atoms', np.nan),
                    "Atoms_Area": results['clean_graphene'].get('total_atoms_area_nm2', np.nan),
                    "Atoms_Density": results['clean_graphene'].get('atoms_density_nm2', np.nan),
                    "Atoms": results['clean_graphene'].get('atoms', []),
                    "Calibrated_FOV": current_fov,
                    "Entire_Area_nm2": Entire_Area_nm2,
                    "Circularities": results[results_key]['shape_metrics'].get('circularity', np.nan),
                    "Roundness": results[results_key]['shape_metrics'].get('roundness', np.nan),
                    "Feret_Diameter": results[results_key]['shape_metrics'].get('feret_diameter', np.nan),
                    "Aspect_Ratio": results[results_key]['shape_metrics'].get('aspect_ratio', np.nan)
                })
            elif self.contamination_analysis:
                contamination = results.get('contamination', {})
                clusters_analysis = contamination.get('Clusters_analysis', {})
                atoms_analysis = contamination.get('Atoms_analysis', {})
                
                new_data.update({
                    "Contamination_Area_nm2": contamination.get('total_area_nm2', np.nan),
                    "Number_of_Clusters": clusters_analysis.get('number_of_clusters', np.nan),
                    "Total_Cluster_Area_nm2": clusters_analysis.get('total_area_nm2', np.nan),
                    "Clusters_Density": clusters_analysis.get('clusters_density_in_contamination_nm2', np.nan),
                    "Clusters": clusters_analysis.get('clusters', []),
                    "Num_Atoms": atoms_analysis.get('number_of_atoms', np.nan),
                    "Atoms_Area": atoms_analysis.get('total_atoms_area_nm2', np.nan),
                    "Atoms_Density": atoms_analysis.get('atoms_density_in_contamination_nm2', np.nan),
                    "Atoms": atoms_analysis.get('atoms', []),
                    "Calibrated_FOV": current_fov,
                    "Entire_Area_nm2": Entire_Area_nm2,
                    "Circularities": results[results_key]['shape_metrics'].get('circularity', np.nan),
                    "Roundness": results[results_key]['shape_metrics'].get('roundness', np.nan),
                    "Feret_Diameter": results[results_key]['shape_metrics'].get('feret_diameter', np.nan),
                    "Aspect_Ratio": results[results_key]['shape_metrics'].get('aspect_ratio', np.nan)
                })

            # 6. Add filter parameters to CSV data
            DEFAULTS = {
                "Gamma": 1.0,
                "CLAHE_Clip": 1.0,
                "CLAHE_Tile": 16,
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
                "Black_Hat": 0,
                "Top_Hat": 0,
                "Circularity": 0.05,
                "Opening2": 0,
                "Closing2": 0,
                "Dilation2": 0,
                "Erosion2": 0,
                "Gradient2": 0,
                "Boundary2": 0,
                "Isolation Distance": 0.05,
                "Brightness": 0.0,
                "Sigmoid Alpha": 1.0,
                "Sigmoid Beta": 1.0,
                "Exp Transform": False,
                "Log Transform": False,
                "Contour_Retrieval_Mode": "RETR_EXTERNAL",
                "Contour_Approximation_Method": "CHAIN_APPROX_TC89_KCOS",
            }

            filter_params = {}
            for param, widget in [
                ("Threshold", self.threshold_slider),
                ("Threshold_SA", self.threshold_slider_sa),
                ("Gamma", self.gamma_slider),
                ("CLAHE_Clip", self.clahe_clip_slider),
                ("CLAHE_Tile", self.clahe_tile_slider),
                ("Gaussian_Sigma", self.gaussian_sigma_slider),
                ("Percentile Contrast", self.contrast_slider),
                ("Double_Gaussian_Sigma1", self.double_gaussian_slider1),
                ("Double_Gaussian_Sigma2", self.double_gaussian_slider2),
                ("Double_Gaussian_Weight", self.double_gaussian_weight_slider),
                ("Dilation", self.dilation_slider),
                ("Erosion", self.erosion_slider),
                ("Opening", self.opening_slider),
                ("Closing", self.closing_slider),
                ("Gradient", self.gradient_slider),
                ("Boundary", self.boundary_slider),
                ("Black_Hat", self.black_hat_slider),
                ("Top_Hat", self.top_hat_slider),
                ("Opening2", self.opening_slider2),
                ("Closing2", self.closing_slider2),
                ("Dilation2", self.dilation_slider2),
                ("Erosion2", self.erosion_slider2),
                ("Gradient2", self.gradient_slider2),
                ("Boundary2", self.boundary_slider2),
                ("Circularity", self.circularity_slider),
                ("Isolation Distance", self.min_isolation_slider),
                ("Make Circular threshold", self.make_circular_thresh),
                ("SA_Cluster_Definer", self.single_atom_clusters_definer),
                ("Brightness", self.brightness_slider),
                ("Sigmoid Alpha", self.sigmoid_alpha),
                ("Sigmoid Beta", self.sigmoid_beta),
                ("Exp Transform", self.exp_transform_checkbox),
                ("Log Transform", self.log_transform_checkbox),
                ("Min_Clean_Cont_Area_nm2", self.min_area_slider),
                ("Max_Clean_Cont_Area_nm2", self.max_area_slider),
                ("Min_Cluster_Area_nm2", self.min_area_sa_clusters),
                ("Max_Cluster_Area_nm2", self.max_area_sa_clusters),
                ("Contour_retrieval_modes", self.contour_retrieval_dropdown),
                ("Contour_approximation_methods", self.contour_approximation_dropdown)
            ]:
                value = widget.value
                default = DEFAULTS.get(param)
                if default is not None:
                    if isinstance(default, float):
                        if np.isclose(value, default, rtol=1e-2):
                            value = np.nan
                    elif isinstance(default, str):
                        if value == default:
                            value = value
                    else:
                        if value == default:
                            value = np.nan
                filter_params[param] = value

            new_data.update(filter_params)
            print("Filter parameters:", filter_params)

        except Exception as e:
            print(f"Error during save: {str(e)}")
            import traceback
            traceback.print_exc()
            return  # Exit on error

        # 8. CSV Handling
        column_order = [
            "Slice", "Threshold", "Threshold_SA", "Contour_retrieval_modes","Contour_approximation_methods",  studied_area, "Number_of_Clusters", 
            "Total_Cluster_Area_nm2", "Clusters_Density", "Clusters", "Num_Atoms", "Atoms_Area", 
            "Atoms_Density", "Atoms", "Min_Clean_Cont_Area_nm2", "Max_Clean_Cont_Area_nm2", 
            "Min_Cluster_Area_nm2", "Max_Cluster_Area_nm2", 
            "Circularities", "Roundness", "Feret_Diameter", "Aspect_Ratio",
            "Calibrated_FOV", "Entire_Area_nm2", 
            "Make Circular threshold", "SA_Cluster_Definer", "Circularity", "Isolation Distance", 
            "Gamma", "Brightness", "Percentile Contrast", "CLAHE_Clip", "CLAHE_Tile", 
            "Sigmoid Alpha", "Sigmoid Beta", "Exp Transform", "Log Transform",
            "Gaussian_Sigma", "Double_Gaussian_Sigma1", "Double_Gaussian_Sigma2", "Double_Gaussian_Weight", 
            "Dilation", "Erosion", "Opening", "Closing", "Gradient", 
            "Opening2", "Closing2", "Dilation2", "Erosion2", "Gradient2",
             ]

        try:
            new_df = pd.DataFrame([new_data])
            
            if os.path.isfile(filename):
                existing_df = pd.read_csv(filename)
                # Preserve all existing columns
                all_columns = list(set(column_order + list(existing_df.columns)))
                existing_df = existing_df.reindex(columns=all_columns)
                new_df = new_df.reindex(columns=all_columns)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df[column_order].to_csv(filename, index=False)
            else:

                new_df = new_df.reindex(columns=column_order)
                new_df.to_csv(filename, index=False)

        except Exception as e:
            print(f"\n=== CSV SAVE ERROR ===")
            print(f"Failed to save CSV: {str(e)}")
            import traceback
            traceback.print_exc()




    
    def _interactive_image_analysis(self):
        if self.analysing_features:
            output = interactive_output(self.update, {'slice_number': self.slice_slider, 'kernel': self.kernel_size_slider,
                                                            'threshold': self.threshold_slider, 'threshold_sa': self.threshold_slider_sa,
                                                            'gamma': self.gamma_slider, 'clahe_clip': self.clahe_clip_slider, 'clahe_tile': self.clahe_tile_slider,
                                                            'gaussian_sigma': self.gaussian_sigma_slider, 'contrast': self.contrast_slider,
                                                            'min_clean_cont_area': self.min_area_slider, 'max_clean_cont_area': self.max_area_slider, 
                                                            'min_cluster_area': self.min_area_sa_clusters, 'max_cluster_area': self.max_area_sa_clusters,
                                                            'double_gaussian_sigma1': self.double_gaussian_slider1, 'double_gaussian_sigma2': self.double_gaussian_slider2,
                                                            'double_gaussian_weight': self.double_gaussian_weight_slider, 
                                                            'iteration_opening': self.opening_slider, 
                                                            'iteration_erosion': self.erosion_slider,
                                                            'iteration_dilation': self.dilation_slider,
                                                            'iteration_closing': self.closing_slider,
                                                            'iteration_gradient': self.gradient_slider,
                                                            'iteration_boundary': self.boundary_slider,
                                                            'iteration_black_hat': self.black_hat_slider,
                                                            'min_circularity': self.circularity_slider,
                                                            'isolation_distance': self.min_isolation_slider,
                                                            'opening2': self.opening_slider2, 
                                                            'closing2': self.closing_slider2,
                                                            'dilation2': self.dilation_slider2,
                                                            'erosion2': self.erosion_slider2,
                                                            'gradient2': self.gradient_slider2,
                                                            'boundary2': self.boundary_slider2,
                                                            'sa_cluster_definer': self.single_atom_clusters_definer,
                                                            'make_circular': self.make_circular_thresh,
                                                            'brightness': self.brightness_slider,
                                                            'sigmoid_alpha': self.sigmoid_alpha,
                                                            'sigmoid_beta': self.sigmoid_beta,
                                                            'exp_transform': self.exp_transform_checkbox,
                                                            'log_transform': self.log_transform_checkbox,
                                                            'contour_retrieval_modes': self.contour_retrieval_dropdown,
                                                            'contour_approximation_methods':self.contour_approximation_dropdown,
                                                            'iteration_top_hat': self.top_hat_slider, 'colormap': self.colormap_dropdown, 'masked_color': self.mask_color_dropdown})
        elif self.save_images_with_calibrated_scalebar:
            output = interactive_output(
            self.display_calibrated_image_with_scalebar,{'kernel': self.kernel_size_slider,'gamma': self.gamma_slider,'slice_number': self.slice_slider,
                                                            'kernel': self.kernel_size_slider,
                                                            'gamma': self.gamma_slider,
                                                            'clahe_clip': self.clahe_clip_slider,
                                                            'clahe_tile': self.clahe_tile_slider,
                                                            'gaussian_sigma': self.gaussian_sigma_slider,
                                                            'contrast': self.contrast_slider,
                                                            'double_gaussian_sigma1': self.double_gaussian_slider1,
                                                            'double_gaussian_sigma2': self.double_gaussian_slider2,
                                                            'double_gaussian_weight': self.double_gaussian_weight_slider,
                                                            'brightness': self.brightness_slider,
                                                            'sigmoid_alpha': self.sigmoid_alpha,
                                                            'sigmoid_beta': self.sigmoid_beta,
                                                            'exp_transform': self.exp_transform_checkbox,
                                                            'log_transform': self.log_transform_checkbox,
                                                            'scalebar_length': self.scalebar_length_slider,
                                                            'colormap': self.colormap_dropdown,
                                                            'x' : self.region_x,
                                                            'y' : self.region_y,
                                                            'h' : self.region_height,
                                                            'w' : self.region_width,
                                                            'resize_factor': self.resize_factor,
                                                            'resize_method': self.resize_method,
                                                            'opening': self.opening_slider,
                                                            # line and region profile parameters
                                                            'line_x': self.line_x,
                                                            'line_y': self.line_y,
                                                            'line_length': self.line_length,
                                                            'line_width': self.line_width,
                                                            'line_angle': self.line_angle,
                                                            'region_profile_x': self.region_profile_x,
                                                            'region_profile_y': self.region_profile_y,
                                                            'region_profile_width': self.region_profile_width,
                                                            'region_profile_height': self.region_profile_height
                                                        })
        return output
    



if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # Check if an instance already exists (important in Jupyter or IPython)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    directories = ['/home/somar/Desktop/2025/Data for publication/Sample 2438/ADF images',
               '/home/somar/Desktop/2025/Data for publication/Sample 2473/ADF images',
               '/home/somar/Desktop/2025/Data for publication/Sample 2474/ADF images',
               '/home/somar/Desktop/2025/Data for publication/Sample 2475/ADF images']

    # stacks = ['/home/somar/Desktop/2025/Data for publication/Multilayer graphene/Sample 2476/stack.h5']

    font_path = "/home/somar/.fonts/SourceSansPro-Semibold.otf" 

    stacks = ['/home/somar/Desktop/2025/Data for publication/Multilayer graphene/Sample 2476/stack.h5']
    stacks_ssb = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/SSB reconstruction of 4d STEM data/stack_ssbs.h5']
    stacks_ssb1 = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/SSB reconstruction of 4d STEM data/stack.h5']

    stacks_adf = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/ADF images/stack.h5']

    stack = stacks[0]
    s = '/home/somar/test /stack.h5'
    image_analysis = InteractiveImageAnalysis(stacks_ssb1[0], metadata_path=None, analysing_features=False, save_images_with_calibrated_scalebar=True, fixed_length_scalebar=True,
                                            clean_graphene_analysis=True, contamination_analysis=False, clusters_analysis=True , defects_analysis=False, font_path=font_path)
    








        # self.line_profile_widgets = self.add_line_profile_widgets()
    # self.region_profile_widgets = self.add_region_profile_widgets()
    # def get_VGVOACurrent(self):
    #     """Get the VG-VOA current from the metadata"""
    #     if self.metadata is not None:
    #         metadata = self.metadata[f"metadata_{self.ref_image_index:04d}"]
    #         vgvoa_current = self.metadata.get_specific_metadata('BP2_^VGVOACurrent', required_keys=['instrument'], data=metadata)
    #         if len(vgvoa_current) > 0:
    #             vgvoa_current = vgvoa_current[0]
    #         else:
    #             vgvoa_current = 1
    #             print('No VG-VOA current found in the metadata. Careful, using a default value of 1 ms.')
    #     else:
    #         print('No metadata available for this reference image. Please use another image for the FFT calibration.')
    #     return vgvoa_current
    

    # def get_dwell_time(self):
    #     """Get the dwell time from the metadata"""
    #     if self.metadata is not None:
    #         metadata = self.metadata[f"metadata_{self.ref_image_index:04d}"]
    #         dwell_time = self.metadata.get_specific_metadata('pixel_time_us', required_keys=['scan_device_properties'], data=metadata)
    #         if len(dwell_time) > 0:
    #             dwell_time = dwell_time[0]
    #         else:
    #             dwell_time = 1
    #             print('No dwell time found in the metadata. Careful, using a default value of 1 ms.')
    #     else:
    #         print('No metadata available for this reference image. Please use another image for the FFT calibration.')
    #     return dwell_time
    

    # def resize_image(self, image):
    #     """Enhanced resizing with dynamic factor and method selection"""
    #     if self.resize_checkbox.value and self.resize_factor_slider.value != 1.0:
    #         image = image.copy()  # Create a copy to avoid modifying the original
    #         method = cv2.INTER_CUBIC if self.resize_method_dropdown.value == 'Bicubic' else cv2.INTER_LINEAR
    #         new_size = (int(image.shape[1] * self.resize_factor_slider.value),
    #                     int(image.shape[0] * self.resize_factor_slider.value))
    #         return cv2.resize(image, new_size, interpolation=method)
    #     return image
    

    # def add_line_profile_widgets(self):
    #     """Create widgets for line profile parameters"""
    #     self.line_x = IntSlider(min=0, max=1650, value=0, description='Line X:')
    #     self.line_y = IntSlider(min=0, max=1650, value=0, description='Line Y:')
    #     self.line_length = IntSlider(min=1, max=500, value=0, description='Length:')
    #     self.line_width = IntSlider(min=1, max=50, value=0, description='Width:')
    #     self.line_angle = FloatSlider(min=0.2, max=360, value=0, description='Angle:')
    #     self.save_line_profile_button = Button(description="Save Line Profile")
    #     self.save_line_profile_button.on_click(self.save_line_profile)
    #     self.line_profile_name = Text(value='line_profile', description='File name:')
        
    #     return VBox([
    #         HBox([self.line_x, self.line_y]),
    #         HBox([self.line_length, self.line_width, self.line_angle]),
    #         HBox([self.save_line_profile_button, self.line_profile_name]),
    #     ])

    # def add_region_profile_widgets(self):
    #     """Create widgets for region profile parameters"""
    #     self.region_profile_x = IntSlider(min=0, max=1650, value=0, description='Region X:')
    #     self.region_profile_y = IntSlider(min=0, max=1650, value=0, description='Region Y:')
    #     self.region_profile_width = IntSlider(min=1, max=500, value=0, description='Width:')
    #     self.region_profile_height = IntSlider(min=1, max=500, value=0, description='Height:')
    #     self.save_region_profile_button = Button(description="Save Region Data")
    #     self.save_region_profile_button.on_click(self.save_region_profile)
    #     self.region_profile_name = Text(value='region_profile', description='File name:')
        
    #     return VBox([
    #         HBox([self.region_profile_x, self.region_profile_y]),
    #         HBox([self.region_profile_width, self.region_profile_height]),
    #         HBox([self.save_region_profile_button, self.region_profile_name]),])
    

    # def get_line_profile(self, image, x, y, length, width, angle):
    #     """Calculate intensity profile along a line"""
    #     from skimage.measure import profile_line
        
    #     # Convert to image coordinates (row, column)
    #     start = (y, x)  # (row, column)
        
    #     # Calculate end point in image coordinates
    #     angle_rad = np.deg2rad(angle)
    #     dx = length * np.cos(angle_rad)
    #     dy = length * np.sin(angle_rad)
    #     end = (y + dy, x + dx)  # (row, column)
        
    #     # Get profile with specified width
    #     profile = profile_line(image, start, end, 
    #                         linewidth=width, 
    #                         reduce_func=np.mean)
    #     # Considering dwell time and VG-VOA current
    #     dwell_time = self.get_dwell_time()
    #     vgvoa_current = self.get_VGVOACurrent()
    #     profile = profile / ( dwell_time * vgvoa_current)
    #     return profile



    # def get_region_profile(self, image, x, y, w, h):
    #     """Calculate average intensity profile in a region"""
    #     region = image[y:y+h, x:x+w]
    #     # Considering dwell time and VG-VOA current
    #     dwell_time = self.get_dwell_time()
    #     vgvoa_current = self.get_VGVOACurrent()
    #     print(dwell_time, vgvoa_current)
    #     region = region / (dwell_time * vgvoa_current)
    #     print('region', region.mean(), region.shape, type(region))
    #     return region.mean()  # we can also specifiy alonge which axis to get the profile using np.mean(axis=0) or np.mean(axis=1)



    # def save_line_profile(self, b):
    #     """Save line profile plot and data"""
    #     params = {
    #         'image': self.stack.raw_data[self.slice_slider.value],
    #         'x': self.line_x.value,
    #         'y': self.line_y.value,
    #         'length': self.line_length.value,
    #         'width': self.line_width.value,
    #         'angle': self.line_angle.value}
        
    #     profile = self.get_line_profile(**params)
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    #     # Save plot
    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     ax.plot(profile)
    #     ax.set_title(f"Line Intensity Profile\n{params['length']}px length, {params['width']}px width")
    #     ax.set_xlabel("Position along line")
    #     ax.set_ylabel("Intensity")
    #     plt.savefig(f"line_profile_{timestamp}.png")
    #     plt.close()
    #     line_profile_name = self.line_profile_name.value
    #     # Save data
    #     np.savetxt(f"{line_profile_name}.csv", profile, delimiter=",")

    

    # def save_region_profile(self, b):
    #     """Save region profile data and marked image"""
    #     params = {
    #         'image': self.stack.raw_data[self.slice_slider.value],
    #         'x': self.region_profile_x.value,
    #         'y': self.region_profile_y.value,
    #         'w': self.region_profile_width.value,
    #         'h': self.region_profile_height.value}
        
    #     profile = self.get_region_profile(**params)
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
    #     # Save marked image
    #     fig, ax = plt.subplots()
    #     ax.imshow(params['image'], cmap='gray')
    #     rect = plt.Rectangle((params['x'], params['y']), params['w'], params['h'],
    #                         linewidth=2, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)
    #     plt.savefig(f"region_marked_{timestamp}.png")
    #     plt.close()

    #     region_profile_name = f"{self.region_profile_name.value}"

    #     # Save data
    #     np.savetxt(f"{region_profile_name}_w{self.region_profile_width.value}_h{self.region_profile_height.value}.csv", [profile], delimiter=",")