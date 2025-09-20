"""
Feature Analysis Class for Image Processing
Author: Somar Dibeh
Date: 2025-06-24
"""

import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
import matplotlib
matplotlib.use("Qt5Agg")
import os
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
import time
from ipywidgets import interactive_output, HBox, VBox, FloatSlider, IntSlider, Checkbox, Output, Dropdown, Label
from data_loader import DataLoader
from filters import *
from imageanalysis.utils.utils_feature_analysis import *
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import connected_components
import imageanalysis.fft_calibration_class as fc
from imageanalysis.fft_calibration_class import *
from joint_widgets import create_widgets
from multiprocessing import Pool, cpu_count
import hashlib

cv2.ocl.setUseOpenCL(False)


class FeaturesAnalysis():

    INTERPOLATION_MAP = {'Bilinear': cv2.INTER_LINEAR,
                        'Bicubic': cv2.INTER_CUBIC,
                        'Lanczos': cv2.INTER_LANCZOS4,
                        'Nearest': cv2.INTER_NEAREST,
                        'Area': cv2.INTER_AREA,}
    
    def __init__(self, stack_path=None, font_path=None,):
        
        widget_dicts = create_widgets()
        for key, value in widget_dicts.items():
            setattr(self, key, value)
        self.stack_path = stack_path
        self.font_path = font_path
        self.stack = DataLoader(self.stack_path) if self.stack_path else []
        self.metadata = self.stack.raw_metadata if self.stack else {}

        self.pool = Pool(processes=12)  # Using 12 processes for multiprocessing


        self.calibration_factor = None
        self.ref_fov = None
        self.calibration_display = Output()
        self.slice_slider = IntSlider(min=0, max=len(self.stack.raw_data)-1, value=0, step=1, description='Slice', continuous_update=False)
        self.image_selector = Dropdown(options=[(f'Image {i}', i) for i in range(len(self.stack.raw_data))] if self.stack else [], 
                                      value=0, description='Image:')
        self.ref_image_index = self.image_selector.value
        self.update_reference_image({'new': self.ref_image_index}) 

        # Setup FFT calibration (but don't display it yet)
        self.fft_calibration = FFTCalibration(self.stack, self.stack_path, display_fft=False)
        self.fft_calibration.calibration_controls.layout.display = 'none'  

        self.save_image_button.on_click(self.save_current_figure)

        self.setup_observers()
        self.save_button.on_click(self.save_data)

        self.manual_thresh_container = HBox(())
        self.kmeans_container = HBox(())
        self.otsu_thresh_container = HBox(())
        # K-means cache
        self.kmeans_cache = {
            'input_hash': None,
            'k': None,
            'attempts': None,
            'epsilon': None,
            'result': None}
        
        # Create tabs
        self.tabs = Tab()
        self.tabs.children = [
            VBox([self.image_filtering_tab()], layout={'padding': '5px'}),
            VBox([self.create_region_tab()], layout={'padding': '5px'}),
            VBox([self.create_calibration_tab()], layout={'padding': '5px'}),
            VBox([ self.first_thresholding_tab()], layout={'padding': '5px'}),
            VBox([self.second_thresholding_tab()], layout={'padding': '5px'}),
            VBox([self.feature_analysis_tab()], layout={'padding': '5px'}),
            VBox([self.create_save_tab()])
            ]
        self.tabs.set_title(0, 'Image Enhancement')
        self.tabs.set_title(1, 'Region Selection')
        self.tabs.set_title(2, 'Calibration')
        self.tabs.set_title(3, '1st thresholding step')
        self.tabs.set_title(4, '2nd thresholding step')
        self.tabs.set_title(5, 'Feature Analysis')
        self.tabs.set_title(6, 'Save & Export')

        self.thresh_method_dropdown.observe(self.toggle_threshold_method, names='value')
        self.toggle_threshold_method({'new': self.thresh_method_dropdown.value})

        display(VBox([self.tabs, self._interactive_image_analysis()],layout={ 'width': '1000px', 'padding': '5px'}))


    def image_filtering_tab(self):
        # Define grid children in [checkbox, control] pairs
        grid_children = [
            [self.contrast_checkbox, self.contrast_slider],
            [self.gaussian_checkbox, self.gaussian_sigma_slider],
            [self.double_gaussian_checkbox, VBox([
                self.double_gaussian_slider1, 
                self.double_gaussian_slider2,
                self.double_gaussian_weight_slider
            ])],

            [self.brightness_checkbox, self.brightness_slider],

            [self.kernel_size_checkbox, self.kernel_size_slider],]
        
        for pair in grid_children:
            # Set checkbox layout
            if isinstance(pair[0], Checkbox):
                pair[0].layout.width = 'auto'
                pair[0].layout.margin = '0 15px 0 0'
            
            # Set slider layout
            if isinstance(pair[1], (FloatSlider, IntSlider)):
                pair[1].layout.width = '300px'
                pair[1].layout.flex = '1 1 auto'
            
            # Set container layout
            if isinstance(pair[1], (VBox, HBox)):
                pair[1].layout.width = 'auto'
                pair[1].layout.flex_flow = 'column'
                pair[1].layout.align_items = 'stretch'
                
                # Set child slider layouts
                for child in pair[1].children:
                    if isinstance(child, (FloatSlider, IntSlider)):
                        child.layout.width = '300px'
                        child.layout.flex = '1 1 auto'
        
        # Create grid rows
        grid_rows = [HBox([pair[0], pair[1]], 
                     layout=Layout(
                         justify_content='space-between',
                         align_items='center',
                         width='100%',
                         margin='5px 0'
                     )) for pair in grid_children]

        return VBox([
            HTML("<h4 style='margin-bottom: 15px;'>Image Enhancement</h4>"),
            VBox(grid_rows, layout=Layout(
                width='100%',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px'
            )),
            HTML("<h4 style='margin-top: 20px; margin-bottom: 15px;'>Display Options</h4>"),
            HBox([
                VBox([self.slice_slider], 
                     layout=Layout(width='45%', margin='0 10px 0 0')),
                VBox([self.colormap_dropdown], 
                     layout=Layout(width='45%', margin='0 0 0 10px'))
            ], layout=Layout(
                width='100%',
                justify_content='space-between',
                margin='10px 0'
            ))
        ], layout=Layout(padding='15px'))


    def create_region_tab(self):
        # Configure widget sizes
        self.region_x_slider.layout.width = '300px'
        self.region_y_slider.layout.width = '300px'
        self.region_width_text.layout.width = '300px'
        # self.region_height_text.layout.width = '300px'
        self.specific_region_checkbox.layout.margin = '0 20px 10px 0'
        self.display_specific_region_checkbox.layout.margin = '0 0 10px 0'

        # Top row: two checkboxes side by side
        checkbox_row = HBox(
            [self.specific_region_checkbox, self.display_specific_region_checkbox],
            layout=Layout(justify_content='space-between', width='100%')
        )

        # Sliders and text fields (shared)
        sliders_box = GridBox(
            children=[
                Label(" "), self.region_x_slider,
                Label(" "), self.region_y_slider,
                Label(" "), self.region_width_text,
                # Label(" "), self.region_height_text,
            ],
            layout=Layout(
                grid_template_columns='max-content 1fr',
                grid_gap='10px 15px',
                padding='15px',
                border='1px solid #e0e0e0',
                border_radius='5px',
                width='100%'
            )
        )

        return VBox(
            [
                HTML("<h4 style='margin-bottom: 15px;'>Region Selection</h4>"),
                checkbox_row,
                sliders_box
            ],
            layout=Layout(padding='15px')
        )

    

    def create_calibration_tab(self):
            def toggle_calibration(change):
                if change['new']:
                    self.fft_calibration.calibration_controls.layout.display = ''
                    # Trigger initial calibration
                    self.fft_calibration.fft_calibrate()
                else:
                    self.fft_calibration.calibration_controls.layout.display = 'none'
                    with self.fft_calibration.calibration_display:
                        clear_output(wait=True)
            
            # Use the FFT class's checkbox instead of creating a new one
            self.fft_calibration.calibration_checkbox_fft_calib.observe(toggle_calibration, names='value')
            
            return VBox([
                self.fft_calibration.calibration_checkbox_fft_calib,
                self.fft_calibration.calibration_controls
            ], layout=Layout(padding='10px'))
        

    def first_thresholding_tab(self):
        # Define rows with their widgets
        rows = [
            [self.analysis_type_dropdown],  # Single widget row
            [self.threshold_checkbox, self.threshold_slider1, self.threshold_slider2],
            [self.min_clean_cont_area_checkbox, self.min_clean_cont_area_slider],
            [self.max_clean_cont_area_checkbox, self.max_clean_cont_area_slider],
            [self.dilation_checkbox, self.dilation_slider],
            [self.erosion_checkbox, self.erosion_slider],
            [self.opening_checkbox, self.opening_slider],
            [self.closing_checkbox, self.closing_slider],

        ]
        
        # Process each widget in each row
        for row in rows:
            for widget in row:
                # Apply consistent styling to all widgets
                if isinstance(widget, Checkbox):
                    widget.layout.width = 'auto'
                    widget.layout.margin = '0 15px 0 0'
                elif isinstance(widget, (FloatSlider, IntSlider)):
                    widget.layout.width = '300px'
                    widget.layout.flex = '1 1 auto'
                elif isinstance(widget, Dropdown):
                    widget.layout.width = 'auto'

        # Create grid rows with appropriate layout
        grid_rows = []
        for row in rows:
            if len(row) == 1:
                # Center single widgets
                hbox = HBox(row, layout=Layout(
                    justify_content='center',
                    width='100%',
                    margin='5px 0'
                ))
            else:
                # Space between multiple widgets
                hbox = HBox(row, layout=Layout(
                    justify_content='space-between',
                    align_items='center',
                    width='100%',
                    margin='5px 0'
                ))
            grid_rows.append(hbox)

        return VBox([
            HTML("<h4 style='margin-bottom: 15px;'>1st Morphological Operations</h4>"),
            VBox(grid_rows, layout=Layout(
                width='100%',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px'
            )),
            HTML("<h4 style='margin-top: 20px; margin-bottom: 15px;'>Display Options</h4>"),
            HBox([
                VBox([self.slice_slider], 
                    layout=Layout(width='45%', margin='0 10px 0 0')),
                VBox([self.colormap_dropdown], 
                    layout=Layout(width='45%', margin='0 0 0 10px'))
            ], layout=Layout(
                width='100%',
                justify_content='space-between',
                margin='10px 0'
            ))
        ], layout=Layout(padding='15px'))


    def second_thresholding_tab(self):
        manual_thresh_widgets = [
            self.threshold_sa_checkbox, 
            self.threshold_sa_slider1, 
            self.threshold_sa_slider2
        ]
        kmeans_widgets = [
            self.kmeans_initial_dropdown,
            self.kmeans_clusters_number, 
            self.kmeans_attempts, 
            self.kmeans_epsilon
        ]
        otsu_thresholding_widgets = [
            self.iterative_otsu_classes_number,
            self.iterative_otsu_region_selection
        ]
        # Update existing containers' children
        self.manual_thresh_container.children = tuple(manual_thresh_widgets)
        self.kmeans_container.children = tuple(kmeans_widgets)
        self.otsu_thresh_container.children = tuple(otsu_thresholding_widgets)
        
        rows = [
            [self.thresh_method_dropdown, 
            self.contour_retrieval_dropdown, 
            self.contour_approximation_dropdown],
            self.otsu_thresh_container,
            self.kmeans_container,
            self.manual_thresh_container,
            [self.min_cluster_area_checkbox, self.min_cluster_area_slider],
            [self.max_cluster_area_checkbox, self.max_cluster_area_slider],
            [self.dilation2_checkbox, self.dilation2_slider],
            [self.erosion2_checkbox, self.erosion2_slider],
            [self.opening2_checkbox, self.opening2_slider],
            [self.closing2_checkbox, self.closing2_slider],
        ]

        # Process each widget in each row
        for row in rows:
            # For HBox containers, get their children
            if isinstance(row, HBox):
                widgets = row.children
            else:
                widgets = row
                
            for widget in widgets:
                # Apply consistent styling to all widgets
                if isinstance(widget, Checkbox):
                    widget.layout.width = 'auto'
                    widget.layout.margin = '0 15px 0 0'
                elif isinstance(widget, (FloatSlider, IntSlider)):
                    widget.layout.width = '300px'
                    widget.layout.flex = '1 1 auto'
                elif isinstance(widget, Dropdown):
                    widget.layout.width = 'auto'

        # Create grid rows with appropriate layout
        grid_rows = []
        for row in rows:
            if isinstance(row, HBox):
                # For HBox containers, just use them directly
                hbox = row
            elif len(row) == 1:
                # Center single widgets
                hbox = HBox(row, layout=Layout(
                    justify_content='center',
                    width='100%',
                    margin='5px 0'
                ))
            else:
                # Space between multiple widgets
                hbox = HBox(row, layout=Layout(
                    justify_content='space-between',
                    align_items='center',
                    width='100%',
                    margin='5px 0'
                ))
            grid_rows.append(hbox)

        return VBox([
            HTML("<h4 style='margin-bottom: 15px;'>2nd Morphological Operations</h4>"),
            VBox(grid_rows, layout=Layout(
                width='100%',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px'
            )),
            HTML("<h4 style='margin-top: 20px; margin-bottom: 15px;'>Display Options</h4>"),
            HBox([
                VBox([self.slice_slider], 
                    layout=Layout(width='45%', margin='0 10px 0 0')),
                VBox([self.colormap_dropdown], 
                    layout=Layout(width='45%', margin='0 0 0 10px'))
            ], layout=Layout(
                width='100%',
                justify_content='space-between',
                margin='10px 0'
            ))
        ], layout=Layout(padding='15px'))


    def feature_analysis_tab(self):
        # Define grid children in [checkbox, control] pairs
        grid_children = [
            [self.feature_analysis_type_dropdown],
            [self.min_isolation_checkbox, self.min_isolation_slider],
            [self.min_circularity_checkbox, self.min_circularity_slider],
            [self.make_circular_thresh_checkbox, self.make_circular_thresh_slider],
            [self.single_atom_clusters_definer_checkbox, self.single_atom_clusters_definer_slider],
        ]
        
        for pair in grid_children:
            if len(pair) > 1:  
                # Set checkbox layout
                if isinstance(pair[0], Checkbox):
                    pair[0].layout.width = 'auto'
                    pair[0].layout.margin = '0 15px 0 0'
                
                # Set slider layout
                if isinstance(pair[1], (FloatSlider, IntSlider)):
                    pair[1].layout.width = '300px'
                    pair[1].layout.flex = '1 1 auto'
                
                # Set container layout
                if isinstance(pair[1], (VBox, HBox)):
                    pair[1].layout.width = 'auto'
                    pair[1].layout.flex_flow = 'column'
                    pair[1].layout.align_items = 'stretch'
                    
                    # Set child slider layouts
                    for child in pair[1].children:
                        if isinstance(child, (FloatSlider, IntSlider)):
                            child.layout.width = '300px'
                            child.layout.flex = '1 1 auto'
        
        # Create grid rows
        grid_rows = [HBox([pair[0], pair[1]], 
                     layout=Layout(
                         justify_content='space-between',
                         align_items='center',
                         width='100%',
                         margin='5px 0'
                     )) for pair in grid_children if len(pair) > 1]

        return VBox([
            HTML("<h4 style='margin-bottom: 15px;'>Features type</h4>"),
            VBox(grid_children[0], layout=Layout(
                width='100%',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px'
            )),
            HTML("<h4 style='margin-bottom: 15px;'>Feature Analysis Control</h4>"),
            VBox(grid_rows, layout=Layout(
                width='100%',
                padding='10px',
                border='1px solid #e0e0e0',
                border_radius='5px'
            )),
            HTML("<h4 style='margin-top: 20px; margin-bottom: 15px;'>Display Options</h4>"),
            HBox([
                VBox([self.slice_slider], 
                     layout=Layout(width='45%', margin='0 10px 0 0')),
                VBox([self.colormap_dropdown], 
                     layout=Layout(width='45%', margin='0 0 0 10px'))
            ], layout=Layout(
                width='100%',
                justify_content='space-between',
                margin='10px 0'
            ))
        ], layout=Layout(padding='15px'))


    def create_save_tab(self):
        return VBox([
            HTML("<h4 style='margin-bottom: 15px;'>Save & Export</h4>"),
            
            # Row 1: number_of_layers_text + number_of_layers_button
            HBox([
                self.number_of_layers_text,
                self.number_of_layers_button
            ], layout=Layout(justify_content='space-between', width='80%', margin='0 0 10px 0')),
            
            # Row 2: image_name + filename_input
            HBox([
                self.image_name,
                self.save_image_button

            ], layout=Layout(justify_content='space-between', width='80%', margin='0 0 10px 0')),
            
            # Row 3: save_image_button + save_button
            HBox([                
                self.filename_input,
                self.save_button
            ], layout=Layout(justify_content='space-between', width='80%', margin='0 0 10px 0')),
            
            # Optional calibration display
            self.calibration_display

        ], layout=Layout(padding='15px'))




    def setup_observers(self):
        # Map checkboxes to their controlled widgets
        controls_map = {
            self.contrast_checkbox: [self.contrast_slider],
            self.gaussian_checkbox: [self.gaussian_sigma_slider],
            self.double_gaussian_checkbox: [
                self.double_gaussian_slider1, 
                self.double_gaussian_slider2,
                self.double_gaussian_weight_slider
            ],
            self.brightness_checkbox: [self.brightness_slider],
            self.kernel_size_checkbox: [self.kernel_size_slider],
            self.threshold_checkbox: [self.threshold_slider1, self.threshold_slider2],
            self.dilation_checkbox: [self.dilation_slider],
            self.erosion_checkbox: [self.erosion_slider],
            self.opening_checkbox: [self.opening_slider],
            self.closing_checkbox: [self.closing_slider],
            self.boundary_checkbox: [self.boundary_slider],
            self.gradient_checkbox: [self.gradient_slider],
            self.threshold_sa_checkbox: [self.threshold_sa_slider1, self.threshold_sa_slider2],
            self.dilation2_checkbox: [self.dilation2_slider],
            self.erosion2_checkbox: [self.erosion2_slider],
            self.opening2_checkbox: [self.opening2_slider],
            self.closing2_checkbox: [self.closing2_slider],
            self.boundary2_checkbox: [self.boundary2_slider],
            self.gradient2_checkbox: [self.gradient2_slider],
            self.min_isolation_checkbox: [self.min_isolation_slider],
            self.min_clean_cont_area_checkbox: [self.min_clean_cont_area_slider],
            self.max_clean_cont_area_checkbox: [self.max_clean_cont_area_slider],
            self.min_cluster_area_checkbox: [self.min_cluster_area_slider],
            self.max_cluster_area_checkbox: [self.max_cluster_area_slider],
            self.min_circularity_checkbox: [self.min_circularity_slider],
            self.make_circular_thresh_checkbox: [self.make_circular_thresh_slider],
            self.single_atom_clusters_definer_checkbox: [self.single_atom_clusters_definer_slider],
            self.specific_region_checkbox: [
                self.region_x_slider, 
                self.region_y_slider,
                self.region_width_text,
                # self.region_height_text
            ],
            self.display_specific_region_checkbox: [
                self.region_x_slider, 
                self.region_y_slider,
                self.region_width_text,
                # self.region_height_text
            ],
        }

        # Set initial visibility
        for checkbox, widgets in controls_map.items():
            for widget in widgets:
                try:
                    widget.layout.display = '' if checkbox.value else 'none'
                except AttributeError:
                    print(f"Warning: Widget {widget} not found")
        
        # Create observers
        for checkbox, widgets in controls_map.items():
            checkbox.observe(
                lambda change, w=widgets: self.toggle_visibility(change, w), 
                names='value'
            )

    def toggle_visibility(self, change, widgets):
        """Toggle widget visibility based on checkbox state"""
        display_value = '' if change['new'] else 'none'
        for widget in widgets:
            try:
                widget.layout.display = display_value
            except AttributeError:
                print(f"Warning: Could not set visibility for {widget}")


    def toggle_threshold_method(self, change):
        method = change['new']
        if method == 'Manual':
            self.manual_thresh_container.layout.display = ''
            self.kmeans_container.layout.display = 'none'
            self.otsu_thresh_container.layout.display = 'none'
        elif method == 'K-means':
            self.manual_thresh_container.layout.display = 'none'
            self.kmeans_container.layout.display = ''
            self.otsu_thresh_container.layout.display = 'none'
        elif method == 'Iterative Otsu':
            self.manual_thresh_container.layout.display = 'none'
            self.kmeans_container.layout.display = 'none'
            self.otsu_thresh_container.layout.display = ''


    def update_reference_image(self, change):
        """Update reference image when selector changes"""
        self.ref_image_index = change['new']
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]




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



    # # Morphological operations
    def apply_morphological_operations(self, thresh, iteration_opening, iteration_closing, iteration_dilation, iteration_erosion, 
                                       iteration_gradient, iteration_boundary, kernel):
        if self.erosion_checkbox.value:
            thresh = erode(thresh, iteration_erosion, kernel)
        if self.dilation_checkbox.value:
            thresh = dilate(thresh, iteration_dilation, kernel)
        if self.opening_checkbox.value:
            thresh = opening(thresh, iteration_opening, kernel)
        if self.closing_checkbox.value:
            thresh = closing(thresh, iteration_closing, kernel)
        if self.gradient_checkbox.value:
            thresh= gradient(thresh, iteration_gradient, kernel)
        if self.boundary_checkbox.value:
            thresh = boundary_extraction(thresh, iteration_boundary, kernel)

        return thresh
    


    def apply_morphological_operations2(self, image, opening2, closing2, dilation2, erosion2, gradient2, boundary2, kernel):
        if self.erosion2_checkbox.value:
            image = erode(image, erosion2, kernel)
        if self.dilation2_checkbox.value:
            image = dilate(image, dilation2, kernel)
        if self.opening2_checkbox.value:
            image = opening(image, opening2, kernel)
        if self.closing2_checkbox.value:
            image = closing(image, closing2, kernel)
        if self.gradient2_checkbox.value:
            image = gradient(image, gradient2, kernel)
        if self.boundary2_checkbox.value:
            image = boundary_extraction(image, boundary2, kernel)
        return image
    


    def apply_filters(self, image, gamma, clahe_clip, clahe_tile, contrast, brightness, sigmoid_alpha, sigmoid_beta):
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply contrast enhancement
        if self.contrast_checkbox.value:
            image = improve_contrast(image, contrast)  #Global contrast correction across the image

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gamma correction (for brightness adjustment)
         #Local contrast correction across each tile of the image
        if self.clahe_checkbox.value:
            image = apply_clahe(image, clahe_clip, clahe_tile)

        if self.gamma_checkbox.value:
            image = apply_gamma_correction(image, gamma)

        # New filters
        if self.brightness_checkbox.value:
            image = cv2.add(image, brightness)

        if self.sigmoid_checkbox.value:
            image = apply_sigmoid_contrast(image, sigmoid_alpha, sigmoid_beta)

        if self.log_transform_checkbox.value:
            image = apply_log_transform(image)

        if self.exp_transform_checkbox.value:
            image = apply_exp_transform(image, gamma)

        return image
    

    def apply_gaussian_double_gaussian(self, image, gaussian_sigma, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel):
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if self.gaussian_checkbox.value:
            kernel_tuple = (kernel, kernel)  # Converting integer kernel size to tuple
            image = cv2.GaussianBlur(image, kernel_tuple, gaussian_sigma)

        # Apply Double Gaussian Filter
        if self.double_gaussian_checkbox.value:
            image = double_gaussian(image, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight)
        return image


    def calculate_isolation(self, contours, nm_per_pixel, isolation_distance):
        """Optimized isolation check with spatial partitioning + multiprocessing"""
        from scipy.spatial import KDTree
        n = len(contours)
        isolation_mask = np.ones(n, dtype=bool)

        if n < 2:
            return isolation_mask

        isolation_px = max(isolation_distance / nm_per_pixel, 0.1)

        # Create bounding circles and spatial tree
        circles = [cv2.minEnclosingCircle(c) for c in contours]
        centers = np.array([(x, y) for (x, y), _ in circles])
        radii = np.array([r for _, r in circles])

        tree = KDTree(centers)
        pairs = list(tree.query_pairs(np.max(radii) * 2 + isolation_px * 2))

        # Use multiprocessing for pair checks
        # I am using 12 processes, you can adjust this based on your CPU cores using cpu_count() instead
        results = self.pool.starmap(
            process_pair,
            [
                (pair, contours, centers, radii, nm_per_pixel, isolation_px, contour_min_distance)
                for pair in pairs
                ]
            )

        for result in results:
            if result is not None:
                i, j = result
                isolation_mask[i] = False
                isolation_mask[j] = False

        return isolation_mask


    def compute_all_areas(self,contours, nm2_per_pixel2):

        areas = self.pool.starmap(compute_area, [(cnt, nm2_per_pixel2) for cnt in contours])
        return areas


    def compute_all_shape_metrics(self, valid_contours, nm_per_pixel):
        args = [
            (cnt, nm_per_pixel,
            measure_circularity,
            measure_roundness,
            measure_feret_diameter,
            measure_aspect_ratio)
            for cnt in valid_contours
        ]

        results = self.pool.map(compute_shape_metrics, args)

        # Extract each metric
        circularity = [r['circularity'] for r in results]
        roundness = [r['roundness'] for r in results]
        feret_diameter = [r['feret_diameter'] for r in results]
        aspect_ratio = [r['aspect_ratio'] for r in results]
        return circularity, roundness, feret_diameter, aspect_ratio
    

    def get_histogram_peaks_info(self, image):
        from scipy.signal import find_peaks
        """
        Get histogram peaks and their properties from the image.
        Returns:
            peaks: List of peak values.
            properties: Dictionary with peak properties like height, width, etc.
        """
        hist, bin_edges = np.histogram(image.flatten(), bins=256, range=(0, 255))
        peaks, _ = find_peaks(hist)
        
        # Calculate properties for each peak
        properties = {
            'peak_values': hist[peaks],
            'peak_indices': peaks,
            'bin_edges': bin_edges[peaks]
        }
        
        return peaks, properties



    def kmeans_thresholding(self, img, k=2, attempts=10,
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
                            flags=cv2.KMEANS_RANDOM_CENTERS):
        """
        Apply K-means clustering to threshold a grayscale image.

        Parameters:
        -----------
        img : np.ndarray
            Grayscale input image (dtype=np.uint8 or convertible).
        k : int
            Number of clusters (default 2 for binary thresholding).
        attempts : int
            Number of times the algorithm is executed using different initial labellings.
        criteria : tuple
            Termination criteria of the algorithm.
        flags : int
            Flags specifying how initial centers are taken.

        Returns:
        --------
        thresholded_img : np.ndarray
            Binary thresholded image (dtype=np.uint8, values 0 or 255).
        """
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Flatten image pixels and convert to float32
        pixel_values = img.reshape((-1, 1)).astype(np.float32)

        # Apply k-means clustering
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, flags)
        # Flatten for easier indexing
        labels = labels.flatten()
        centers = centers.flatten()
        # Sort centers and get mapping from old index → new sorted index
        sort_idx = np.argsort(centers)
        centers_sorted = centers[sort_idx]

        # Remap labels to match sorted centers
        labels_sorted = np.zeros_like(labels)
        for new_label, old_label in enumerate(sort_idx):
            labels_sorted[labels == old_label] = new_label

        # Compute thresholds between consecutive sorted centers
        thresholds_list = [(centers_sorted[i] + centers_sorted[i + 1]) / 2
                        for i in range(len(centers_sorted) - 1)]

        # Identify foreground cluster: cluster with max center intensity
        foreground_cluster = np.argmax(centers)

        # Prepare output binary image (flattened for indexing)
        thresholded_img = np.zeros_like(labels, dtype=np.uint8)
        thresholded_img[labels.flatten() == foreground_cluster] = 255

        # Reshape back to original image shape
        thresholded_img = thresholded_img.reshape(img.shape)

        return thresholded_img, thresholds_list
    
    

    def get_kmeans_threshold(self, masked_clean_image, k_number, attempts_number, epsilon, kmeans_init=cv2.KMEANS_RANDOM_CENTERS):
        import hashlib
        current_hash = hashlib.md5(masked_clean_image.tobytes()).hexdigest()
        
        # Cached result available
        if (self.kmeans_cache['input_hash'] == current_hash and
            self.kmeans_cache['kmeans_init'] == kmeans_init and
            self.kmeans_cache['k'] == k_number and
            self.kmeans_cache['attempts'] == attempts_number and
            self.kmeans_cache['epsilon'] == epsilon and
            self.kmeans_cache['result'] is not None):
            return self.kmeans_cache['result']  # <-- now result will be a tuple

        # Compute fresh
        np.random.seed(40)
        thresh_sa, thresholds_list = self.kmeans_thresholding(
            masked_clean_image, 
            k=k_number, 
            attempts=attempts_number, 
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, epsilon), 
            flags=kmeans_init
        )

        # Cache *both* values
        self.kmeans_cache = {
            'input_hash': current_hash,
            'kmeans_init': kmeans_init,
            'k': k_number,
            'attempts': attempts_number,
            'epsilon': epsilon,
            'result': (thresh_sa, thresholds_list)  # tuple stored
        }

        return thresh_sa, thresholds_list


    def resize_image(self, image, factor, method):
        import cv2
        if factor == 1.0:
            return image  # No resizing needed

        interpolation_method = self.INTERPOLATION_MAP.get(method, cv2.INTER_LINEAR)
        new_size = (int(image.shape[1] * factor), int(image.shape[0] * factor))
        resized_image = cv2.resize(image, new_size, interpolation=interpolation_method)
        return resized_image



    def get_valid_contours(self, image, threshold1, threshold2, clean_graphene_analysis, min_area, max_area, iteration_opening, 
                           iteration_closing, iteration_dilation, iteration_erosion, iteration_gradient, iteration_boundary, kernel, nm2_per_pixel2):
        
        thresh = cv2.inRange(image, threshold1, threshold2) if clean_graphene_analysis else ~cv2.inRange(image, threshold1, threshold2)
        thresh = thresh.astype(np.uint8)
        percentile_threshold1, percentile_threshold2 = np.sum(image<=threshold1) / image.size * 100, np.sum(image<=threshold2) / image.size * 100
        percentile_grey_values1, percentile_grey_values2 = threshold1 / 255 * 100, threshold2 / 255 * 100
        print(f"1st threshold value: {threshold1}, Percentile area: {percentile_threshold1:.2f}%, Percentile grey values: {percentile_grey_values1:.2f}%")
        print(f"2nd threshold value: {threshold2}, Percentile area: {percentile_threshold2:.2f}%, Percentile grey values: {percentile_grey_values2:.2f}%")
        thresh = self.apply_morphological_operations(thresh, iteration_opening, iteration_closing, iteration_dilation,
                                                    iteration_erosion, iteration_gradient, iteration_boundary, kernel)
        # Clean area detection
        contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        valid_contours = [cnt for cnt in contours if min_area <= (cv2.contourArea(cnt) * nm2_per_pixel2) < max_area]
        return valid_contours, thresh



    def analyse_features(self, threshold1, threshold2, threshold_sa1, threshold_sa2, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast, min_clean_cont_area, min_cluster_area,
            max_cluster_area, max_clean_cont_area, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, 
            iteration_opening, iteration_closing, iteration_dilation, iteration_erosion, iteration_gradient, min_circularity, isolation_distance,
            iteration_boundary, opening2, closing2, dilation2, erosion2, gradient2, boundary2, sa_cluster_definer,
            brightness, sigmoid_alpha, sigmoid_beta, exp_transform, log_transform, make_circular,
            contour_retrieval_modes, contour_approximation_methods, x, y, w, kmeans_init, attempts_number, k_number, epsilon, otsu_classes_number, 
            otsu_region_selection, colormap=None, slice_number=None, kernel=None, masked_color=None, display_images=True):
        """
        isolation_distance: Minimum pixel distance between atoms to consider them isolated
        The image is saved in many variables name for a purpose:
        img: original image after normalization that undergoes only (Gaussian, Double Gaussian) filtering and that is used for the analysis
        image: image used for display that undergoes all the filters selected
        cropped_image: cropped image used for display when display specific region is selected and for analysis when select specific region checkbox is selected
        norm_image: normalized original image used for display when no filters are selected
        """
        import cv2
        from scipy.signal import find_peaks
        from skimage import filters
        with self.calibration_display:
            clear_output(wait=True)
        try:
            original_image = self.stack.raw_data[slice_number]
            nm_per_pixel, nm2_per_pixel2 = self.fft_calibration.get_calibrated_image(original_image, slice_number)
            if self.metadata.key_in('pixel_time_us', data=self.metadata[slice_number]) is True:
                print('True')
                pixel_time_us = self.metadata.get_specific_metadata("pixel_time_us", required_keys=['scan_device_properties'])[slice_number]
                print(f"Pixel time in microseconds: {pixel_time_us}")
            else:
                print("Pixel time information not available in metadata.")
            # Avoiding complex images
            if np.iscomplexobj(original_image):
                img_in = np.real(original_image)  # magnitude for complex
            else:
                img_in = original_image          # keep real values (even negative)
            img = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            norm_image = img.copy()

            # Cropping image and investigating specific area
            if self.specific_region_checkbox.value or self.display_specific_region_checkbox.value:
                # Important: we need square image to get the calibration correct (if you want to use rectangular images, you need to adjust the get_calibrated_image method in 
                # the fft_calibration class accordingly otherwise you will get incorrect results)
                x_adj = int(x)
                y_adj = int(y)
                w_adj = int(w)
                h_adj = w_adj
                rect = patches.Rectangle((x_adj, y_adj), w_adj, h_adj, linewidth=1, 
                        edgecolor='red', facecolor='none')
                
            if self.specific_region_checkbox.value:
                img = img[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
                nm_per_pixel, nm2_per_pixel2 = self.fft_calibration.get_calibrated_image(img, slice_number)
                print('resulting_fov after cropping:', nm_per_pixel*img.shape[0])
                cropped_image = img.copy()
                cropped_image = self.apply_filters(cropped_image, gamma, clahe_clip, clahe_tile, contrast, brightness, sigmoid_alpha, sigmoid_beta)


            if self.gaussian_checkbox.value or self.double_gaussian_checkbox.value:
                # Apply Gaussian and/or Double Gaussian filters
                img = self.apply_gaussian_double_gaussian(img, gaussian_sigma, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel)
                if self.specific_region_checkbox.value:
                    cropped_image = self.apply_gaussian_double_gaussian(cropped_image, gaussian_sigma, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel)
            # Improve visualization of the image, the filtered image won't be used for analysis rather the original image (img) will be used. This is just to see what we are analyzing
            image = self.apply_filters(norm_image, gamma, clahe_clip, clahe_tile, contrast, brightness, sigmoid_alpha, sigmoid_beta)
            image = self.apply_gaussian_double_gaussian(image, gaussian_sigma, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel)
            peaks, properties = self.get_histogram_peaks_info(img)

            # Determine analysis type from dropdown
            analysis_type = self.analysis_type_dropdown.value
            clean_graphene_analysis = (analysis_type == 'Clean_area_analysis')
            contamination_analysis = (analysis_type == 'Contaminated_area_analysis')
            
            # Determine feature analysis type
            feature_analysis_type = self.feature_analysis_type_dropdown.value
            clusters_sa_analysis = (feature_analysis_type == 'Single_atom_clusters_analysis')
            defects_analysis = (feature_analysis_type == 'Defects_analysis')

            valid_contours, first_thresh = self.get_valid_contours(img, threshold1, threshold2, clean_graphene_analysis, min_clean_cont_area, max_clean_cont_area, 
                                                    iteration_opening, iteration_closing, iteration_dilation, iteration_erosion, iteration_gradient, 
                                                    iteration_boundary, kernel, nm2_per_pixel2)

            contour_mask = np.zeros_like(img).astype(np.uint8)
            cv2.drawContours(contour_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

            # Cluster/atom detection with isolation filtering
            img = img.astype(np.uint8)
            print(f"img dtype: {img.dtype}, shape: {img.shape}")
            print(f"mask dtype: {contour_mask.dtype}, shape: {contour_mask.shape}")

            masked_clean_image = cv2.bitwise_and(img, img, mask=contour_mask)

            # # Otsu for second masking in case of cluster analysis
            # otsu_thresh_value, otsu_mask = cv2.threshold(masked_clean_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # # Optional: filter contours by area
            # contours_otsu, _ = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


            # # Draw valid contours into mask
            # contour_mask_otsu = np.zeros_like(masked_clean_image, dtype=np.uint8)
            # cv2.drawContours(contour_mask_otsu, contours_otsu, -1, 255, thickness=cv2.FILLED)

            # # Apply final mask
            # masked_clean_image= cv2.bitwise_and(masked_clean_image, masked_clean_image, mask=contour_mask_otsu)
            # masked_clean_image = np.where(masked_clean_image < otsu_thresh_value, otsu_thresh_value, masked_clean_image).astype(np.uint8)

            thresholds_list = []
            # Determine second thresholding method
            threshold_method = self.thresh_method_dropdown.value
            manual_thresholding = (threshold_method == 'Manual')
            kmeans_thresholding = (threshold_method == 'K-means')
            iterative_otsu_thresholding = (threshold_method == 'Iterative Otsu')

            from skimage.filters import threshold_multiotsu
            if manual_thresholding:
                if clusters_sa_analysis:
                    thresh_sa = cv2.inRange(masked_clean_image, threshold_sa1, threshold_sa2)
                elif defects_analysis:
                    thresh_sa = ~cv2.inRange(masked_clean_image, threshold_sa1, threshold_sa2)
                else:
                    thresh_sa = np.zeros_like(masked_clean_image, dtype=np.uint8)
                thresholds_list = [threshold_sa1, threshold_sa2]

     
            # Iterative Otsu thresholding
            elif iterative_otsu_thresholding:
                unique_vals = np.unique(masked_clean_image)
                if unique_vals.size >= otsu_classes_number + 1:   # enough values for 4 classes
                    thresholds = threshold_multiotsu(masked_clean_image, classes=otsu_classes_number)
                    regions = np.digitize(masked_clean_image, bins=thresholds)
                    thresholds_list = thresholds.tolist()
                else:
                    thresholds = np.array([])
                    regions = np.zeros_like(masked_clean_image)
                    thresholds_list = []
                regions = np.digitize(masked_clean_image, bins=thresholds)
                if clusters_sa_analysis:
                    # Example: keep only the highest-intensity class (atoms/bright features)
                    thresh_sa = (regions >= otsu_region_selection).astype(np.uint8) * 255
                elif defects_analysis:
                    # Example: keep the lowest-intensity class (dark defects)
                    thresh_sa = (regions < otsu_region_selection).astype(np.uint8) * 255
                else:
                    # Default: keep everything
                    thresh_sa = np.zeros_like(masked_clean_image, dtype=np.uint8)
                for i in range(len(thresholds_list)):
                    percentile_threshold_sa = np.sum(masked_clean_image<=thresholds_list[i]) / masked_clean_image.size * 100
                    percentile_grey_values = thresholds_list[i] / 255 * 100
                    print(f"2nd threshold value: {thresholds_list[i]}, Percentile area: {percentile_threshold_sa:.2f}%, Percentile grey values: {percentile_grey_values:.2f}%")

            # Trying k-means thresholding
            elif kmeans_thresholding:
                kmeans_init = self.kmeans_initialization_methods[kmeans_init]
                thresh_sa, thresholds_list = self.get_kmeans_threshold(masked_clean_image, k_number, attempts_number, epsilon, kmeans_init=kmeans_init)
                
                # Print thresholds here instead of later
                for i in range(len(thresholds_list)):
                    percentile_threshold_sa = np.sum(masked_clean_image<=thresholds_list[i]) / masked_clean_image.size * 100
                    percentile_grey_values = thresholds_list[i] / 255 * 100
                    print(f"2nd threshold value: {thresholds_list[i]}, Percentile area: {percentile_threshold_sa:.2f}%, Percentile grey values: {percentile_grey_values:.2f}%")
            else:
                raise ValueError("Invalid thresholding method selected. Choose 'Manual' or 'K-means'.")

            thresh_sa = thresh_sa.astype(np.uint8)
            thresh_sa = self.apply_morphological_operations2(thresh_sa, opening2, closing2, dilation2, erosion2, gradient2, boundary2, kernel)

            # # Finding clusters and single atoms
            retrieval_mode = self.contour_retrieval_modes[contour_retrieval_modes]
            approx_method = self.contour_approximation_methods[contour_approximation_methods]

            contours_sa, _ = cv2.findContours(thresh_sa, retrieval_mode, approx_method)

            # Convert to integer coordinates
            contours_sa = [c.astype(np.int32) for c in contours_sa]

            # Calculate isolation status for all original SA contours
            if self.min_isolation_checkbox.value:
                isolation_mask = self.calculate_isolation(contours_sa, nm_per_pixel, isolation_distance)
            else:
                isolation_mask = np.ones(len(contours_sa), dtype=bool)

            # Independent condition checks
            valid_contours_sa = []
            centroids = []
            for idx, cnt in enumerate(contours_sa):
                # mask = np.zeros_like(img, dtype=np.uint8)
                # cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
                # area = np.count_nonzero(mask)            # Choosing this will gives the exact area of pixels in the contour and fit the display when changing the area condition slider, but overestimates the area for small contours
                # contour_area = cv2.contourArea(cnt)     # Choosing this will gives the area based on the contour line, which is more accurate for small contours but does not fit the display
                area = cv2.contourArea(cnt, oriented=False)  # getting absolute contour area by setting oriented=False otherwise it returns the signed area
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

            circularity, roundness, feret_diameter, aspect_ratio = self.compute_all_shape_metrics(valid_contours_sa, nm_per_pixel)
            # Prepare data for histograms
            clean_areas = self.compute_all_areas(valid_contours, nm2_per_pixel2)
            cluster_areas = self.compute_all_areas(valid_contours_sa, nm2_per_pixel2)

            print("1st checking:", len(circularity), len(roundness), len(feret_diameter), len(aspect_ratio), len(clean_areas), len(cluster_areas))
        #Results calculation
            if clean_graphene_analysis:
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

            elif contamination_analysis:
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


            histogram_peaks = {
                'peaks': peaks.tolist(),
                'properties': {
                    'peak_values': properties['peak_values'].tolist(),
                    'peak_indices': properties['peak_indices'].tolist(),
                    'bin_edges': properties['bin_edges'].tolist()
                }
            }


            results = {  'histogram_peaks': histogram_peaks,
                        'clean_graphene': {},
                        'contamination': {},
                        'clusters_and_single_atoms': {},
                        'thresholding_list': thresholds_list
                        }

            if clean_graphene_analysis:
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

            elif contamination_analysis:
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
            if clean_graphene_analysis or contamination_analysis:
                results_key = 'clean_graphene' if clean_graphene_analysis else 'contamination'
                results[results_key]['shape_metrics'] = {
                    'circularity': circularity,
                    'roundness': roundness,
                    'feret_diameter': feret_diameter,
                    'aspect_ratio': aspect_ratio
                }

            # Display images if requested 
            area_of_interest_label = "Clean graphene area" if clean_graphene_analysis else "Contamination area"
            area_of_interest = total_clean_area_nm2 if clean_graphene_analysis else total_contamination_area_nm2


            # Visualization

            # Replace small contours with circles of the same area (This is for plotting purposes, it doesn't affect the analysis)
            if self.make_circular_thresh_checkbox.value:
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
            else:
                modified_contours = valid_contours_sa  # to avoid errors if no contours are modified

            if display_images:
                masked_clean_image = masked_clean_image.astype(np.uint8)
                masked_image = cv2.cvtColor(masked_clean_image, cv2.COLOR_GRAY2RGB)
                masked_image[contour_mask == 0] = self.mask_color(masked_color)
                cv2.drawContours(masked_image, valid_contours_sa, -1, (0, 255, 0), 1)

                histogram, bins = np.histogram(img, bins=256, range=(0, 255))
                aoi_histogram, aoi_bins = np.histogram(masked_clean_image, bins=256, range=(0, 255))    # aoi = area of interest
                valid_contour_sa_mask = np.zeros_like(img)
                cv2.drawContours(valid_contour_sa_mask, modified_contours, -1, 255, thickness=cv2.FILLED)

                # Create or reuse figure
                if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
                    self.fig, self.axs = plt.subplots(2, 4, figsize=(20, 10))
                    self.fig.show()
                else:
                    # Clear previous content
                    for ax in self.axs.flatten():
                        ax.cla()

                # Update plots with new data
                self.axs[0,0].imshow(image, cmap=colormap)
                self.axs[0,0].set_title('Filtered Image')
                self.axs[0,0].add_patch(rect) if self.specific_region_checkbox.value or self.display_specific_region_checkbox.value else None

                if self.specific_region_checkbox.value:
                    self.axs[0,1].imshow(cropped_image, cmap='viridis')
                    self.axs[0,1].set_title('Thresholding of original image')
                elif self.display_specific_region_checkbox.value:
                    image = img[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
                    image = self.apply_filters(image, gamma, clahe_clip, clahe_tile, contrast, brightness, sigmoid_alpha, sigmoid_beta)
                    self.axs[0,1].imshow(image, cmap='viridis')
                    self.axs[0,1].set_title('Cropped region of original image')
                else:
                    self.axs[0,1].imshow(first_thresh, cmap='viridis')
                    self.axs[0,1].set_title('Cropped region of original image')

                self.axs[0,2].imshow(contour_mask, cmap=colormap)
                self.axs[0,2].set_title('Filtered thresholded image')

                self.axs[0,3].plot(np.log1p(histogram), color='blue')
                self.axs[0,3].set_ylabel('Count')
                self.axs[0,3].set_title('Histogram of processed image')
                self.axs[0,3].grid(True)    
                self.axs[0,3].set_xlim(0, 25)  
                self.axs[0,3].set_ylim(0, 40)  
                self.axs[0,3].axvline(x=threshold1, color='red', linestyle='--', label=f'Percentile: {threshold1:.2f}%')
                self.axs[0,3].axvline(x=threshold2, color='red', linestyle='--', label=f'Percentile: {threshold2:.2f}%')

                self.axs[0,3].axis('on')
                info_text = (
                    f"Number of {area_of_interest_label} found:   {len(clean_areas)}\n"
                    f"Total {area_of_interest_label}: {area_of_interest:.2f} nm²\n")

                anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=10),
                                            frameon=True, pad=0.5, borderpad=0.5)
                anchored_text.patch.set_boxstyle("round,pad=0.3")
                anchored_text.patch.set_facecolor("white")
                anchored_text.patch.set_alpha(0.9)
                self.axs[0,3].add_artist(anchored_text)
                
                self.axs[1,0].imshow(masked_image, cmap='gray')
                self.axs[1,0].set_title('Contamination Mask keeping only clean graphene')

                self.axs[1,1].imshow(thresh_sa, cmap=colormap)
                self.axs[1,1].set_title('Threholding of the clean graphene area')
                if self.display_specific_region_checkbox.value:
                    valid_contour_sa_mask = valid_contour_sa_mask[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
                    self.axs[1,2].imshow(valid_contour_sa_mask, cmap='gray')
                    self.axs[1,2].set_title('Filtered clusters and single atoms in cropped region')
                else:
                    self.axs[1,2].imshow(valid_contour_sa_mask, cmap='gray')
                    self.axs[1,2].set_title('Filtered clusters and single atoms')

                self.axs[1,3].plot(np.log1p(aoi_histogram), color='green')
                self.axs[1,3].set_title('Histogram of investigated area after first thresholding')
                self.axs[1,3].set_ylabel('Count')
                self.axs[1,3].grid(True)
                self.axs[1,3].set_xlim(0, 60)  
                self.axs[1,3].set_ylim(0, 20)
                for threshold in thresholds_list:
                    self.axs[1,3].axvline(x=threshold, color='red', linestyle='--', label=f'Percentile: {threshold:.2f}%') 

                self.axs[1,3].axis('on')
                info_text = (
                    f"Number of clusters found:  {len(cluster_areas)}\n")
                    # f"Total cluster area in {area_of_interest_label}: {total_sa_cluster_area_nm2:.2f} nm²\n")
                anchored_text = AnchoredText(info_text, loc='upper left', prop=dict(size=10),
                                            frameon=True, pad=0.5, borderpad=0.5)
                anchored_text.patch.set_boxstyle("round,pad=0.3")
                anchored_text.patch.set_facecolor("white")
                anchored_text.patch.set_alpha(0.9)
                self.axs[1,3].add_artist(anchored_text)
                # # Redraw only the changed elements
                for ax in self.axs.flatten():
                    if ax == self.axs[0,3] or ax == self.axs[1,3]:  # Only redraw the first image
                        ax.axis('on')
                    else:
                        ax.axis('off')

                self.fig.tight_layout()
                self.fig.canvas.draw()

            if clean_graphene_analysis or contamination_analysis:
                print(f"Final {results_key} shape metrics: {results[results_key]['shape_metrics']}")
            return results

        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {} 
        


    def save_current_figure(self, b):
        if not hasattr(self, 'fig') or not plt.fignum_exists(self.fig.number):
            print("No figure to save")
            return
        fig_name = f"{self.image_name.value}.svg"
        if not fig_name.endswith('.svg'):
            fig_name += '.svg'
        dirname, image_name = os.path.split(fig_name)
        if dirname == '':
            dirname = '.'
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        self.fig.savefig(fig_name, format='svg')
        print(f"Figure saved as {fig_name}")

        # force filesystem sync
        if os.name == 'nt':  # For Windows
            os.system(f'cmd /c "echo. > {os.path.join(dirname, "refresh.tmp")}"')
            os.remove(os.path.join(dirname, "refresh.tmp"))
        else:  # For Linux/Mac
            os.sync()

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
            nm_per_pixel, _ = self.fft_calibration.get_calibrated_image(image, slice_number)
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
                for col in new_data.keys():
                    if col in df.columns:
                        df.loc[mask, col] = new_data[col]
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            
            # 5. Save back
            df.to_csv(filename, index=False)
            print(f"Updated slice {slice_number}: FOV={calibrated_fov:.2f} nm")
            
        except Exception as e:
            print(f"Save failed: {str(e)}")
            import traceback
            traceback.print_exc()



    def _force_filesystem_refresh(self, directory):
        """Force filesystem refresh in VS Code"""
        temp_file = os.path.join(directory, ".__refresh__")
        
        # Create and immediately delete a temp file
        try:
            with open(temp_file, 'w') as f:
                f.write('')
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not force refresh ({str(e)})")
        
        # Additional sync for Linux/Mac
        if os.name != 'nt':
            os.sync()


    def save_data(self, _):
        import json
        import os
        import pandas as pd
        import numpy as np
        
        # 1. Get filename and ensure directory exists
        base_name = self.filename_input.value
        if not base_name:
            print("Error: Filename is empty")
            return
        
        filename = f"{base_name}.csv" if not base_name.endswith('.csv') else base_name
        dir_path = os.path.dirname(filename)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        slice_number = self.slice_slider.value
        
        # 2. Prepare data storage with default values
        new_data = {
            "Slice": slice_number,
            "Histogram_Peaks": {},
            "Number_of_Layers": self.number_of_layers_text.value,
            "Clean_Area_nm2": np.nan,
            "Contamination_Area_nm2": np.nan,
            "Number_of_Clusters": np.nan,
            "Total_Cluster_Area_nm2": np.nan,
            "Clusters_Density": np.nan,
            "Clusters": "[]",
            "Num_Atoms": np.nan,
            "Atoms_Area": np.nan,
            "Atoms_Density": np.nan,
            "Atoms": "[]",
            "Calibrated_FOV": np.nan,
            "Entire_Area_nm2": np.nan,
            "Circularities": "[]",
            "Roundness": "[]",
            "Feret_Diameter": "[]",
            "Aspect_Ratio": "[]",
            "Thresholds_List_Kmeans": "[]"
        }

        try:
            # Collect parameters for analysis
            params = {
                "threshold1": self.threshold_slider1.value,
                "threshold2": self.threshold_slider2.value,
                "threshold_sa1": self.threshold_sa_slider1.value,
                "threshold_sa2": self.threshold_sa_slider2.value,
                "gamma": self.gamma_slider.value,
                "clahe_clip": self.clahe_clip_slider.value,
                "clahe_tile": self.clahe_tile_slider.value,
                "gaussian_sigma": self.gaussian_sigma_slider.value,
                "contrast": self.contrast_slider.value,
                "min_clean_cont_area": self.min_clean_cont_area_slider.value,
                "max_clean_cont_area": self.max_clean_cont_area_slider.value,
                "min_cluster_area": self.min_cluster_area_slider.value,
                "max_cluster_area": self.max_cluster_area_slider.value,
                "make_circular": self.make_circular_thresh_slider.value,
                "double_gaussian_sigma1": self.double_gaussian_slider1.value,
                "double_gaussian_sigma2": self.double_gaussian_slider2.value,
                "double_gaussian_weight": self.double_gaussian_weight_slider.value,
                "iteration_opening": self.opening_slider.value,
                "iteration_closing": self.closing_slider.value,
                "iteration_dilation": self.dilation_slider.value,
                "iteration_erosion": self.erosion_slider.value,
                "iteration_gradient": self.gradient_slider.value,
                "iteration_boundary": self.boundary_slider.value,
                "min_circularity": self.min_circularity_slider.value,
                "isolation_distance": self.min_isolation_slider.value,
                "opening2": self.opening2_slider.value,
                "closing2": self.closing2_slider.value,
                "dilation2": self.dilation2_slider.value,
                "erosion2": self.erosion2_slider.value,
                "gradient2": self.gradient2_slider.value,
                "boundary2": self.boundary2_slider.value,
                "sa_cluster_definer": self.single_atom_clusters_definer_slider.value,
                "slice_number": slice_number,
                "kernel": self.kernel_size_slider.value,
                "brightness": self.brightness_slider.value,
                "sigmoid_alpha": self.sigmoid_alpha_slider.value,
                "sigmoid_beta": self.sigmoid_beta_slider.value,
                "exp_transform": self.exp_transform_checkbox.value,
                "log_transform": self.log_transform_checkbox.value,
                'contour_retrieval_modes': self.contour_retrieval_dropdown.value,
                'contour_approximation_methods': self.contour_approximation_dropdown.value,
                'x' : self.region_x_slider.value,
                'y' : self.region_y_slider.value,
                'w' : self.region_width_text.value,
                "kmeans_init": self.kmeans_initial_dropdown.value,
                "attempts_number": self.kmeans_attempts.value,
                "k_number": self.kmeans_clusters_number.value,
                "epsilon": self.kmeans_epsilon.value,
                "otsu_classes_number": self.iterative_otsu_classes_number.value,
                "otsu_region_selection": self.iterative_otsu_region_selection.value,
                "display_images": False
            }
            
            # Run analysis
            results = self.analyse_features(**params)
            if 'histogram_peaks' in results:
                new_data['Histogram_Peaks'] = json.dumps(results['histogram_peaks'])
            else:
                print("Warning: No histogram peaks found in results")
            if 'thresholding_list' in results:
                new_data['Thresholds_List_Kmeans'] = json.dumps(results['thresholding_list'])
            # Determine analysis type
            analysis_type = self.analysis_type_dropdown.value
            clean_graphene_analysis = (analysis_type == 'Clean_area_analysis')
            contamination_analysis = (analysis_type == 'Contaminated_area_analysis')
            
            # Determine feature analysis type
            feature_analysis_type = self.feature_analysis_type_dropdown.value
            clusters_sa_analysis = (feature_analysis_type == 'Single_atom_clusters_analysis')
            defects_analysis = (feature_analysis_type == 'Defects_analysis')

            # Get calibration data
            if self.specific_region_checkbox.value:                          
                # Important: we need square image to get the calibration correct (if you want to use rectangular images, you need to adjust the get_calibrated_image method in 
                # the fft_calibration class accordingly otherwise you will get incorrect results)
                x = self.region_x_slider.value
                y = self.region_y_slider.value
                w = self.region_width_text.value
                h = w
                image = self.stack.raw_data[slice_number][y:y+h, x:x+w]
            else:
                image = self.stack.raw_data[slice_number]
            nm_per_pixel, nm2_per_pixel2 = self.fft_calibration.get_calibrated_image(image, slice_number)
            current_fov = image.shape[0] * nm_per_pixel  
            entire_area_nm2 = current_fov ** 2
            
            # Update new_data with calibration
            new_data.update({
                "Calibrated_FOV": current_fov,
                "Entire_Area_nm2": entire_area_nm2
            })
            
            # Handle results based on analysis type
            if clean_graphene_analysis:
                graphene = results.get('clean_graphene', {})
                new_data.update({
                    "Clean_Area_nm2": graphene.get('total_area_nm2', np.nan),
                    "Number_of_Clusters": graphene.get('number_of_clusters', np.nan),
                    "Total_Cluster_Area_nm2": graphene.get('total_cluster_area_nm2', np.nan),
                    "Clusters_Density": graphene.get('clusters_density_in_graphene_nm2', np.nan),
                    "Clusters": json.dumps(graphene.get('clusters', [])),
                    "Num_Atoms": graphene.get('number_of_atoms', np.nan),
                    "Atoms_Area": graphene.get('total_atoms_area_nm2', np.nan),
                    "Atoms_Density": graphene.get('atoms_density_nm2', np.nan),
                    "Atoms": json.dumps(graphene.get('atoms', [])),
                    "Circularities": json.dumps(graphene.get('shape_metrics', {}).get('circularity', [])),
                    "Roundness": json.dumps(graphene.get('shape_metrics', {}).get('roundness', [])),
                    "Feret_Diameter": json.dumps(graphene.get('shape_metrics', {}).get('feret_diameter', [])),
                    "Aspect_Ratio": json.dumps(graphene.get('shape_metrics', {}).get('aspect_ratio', []))
                })
            elif contamination_analysis:
                contamination = results.get('contamination', {})
                clusters_analysis = contamination.get('Clusters_analysis', {})
                atoms_analysis = contamination.get('Atoms_analysis', {})
                
                new_data.update({
                    "Contamination_Area_nm2": contamination.get('total_area_nm2', np.nan),
                    "Number_of_Clusters": clusters_analysis.get('number_of_clusters', np.nan),
                    "Total_Cluster_Area_nm2": clusters_analysis.get('total_area_nm2', np.nan),
                    "Clusters_Density": clusters_analysis.get('clusters_density_in_contamination_nm2', np.nan),
                    "Clusters": json.dumps(clusters_analysis.get('clusters', [])),
                    "Num_Atoms": atoms_analysis.get('number_of_atoms', np.nan),
                    "Atoms_Area": atoms_analysis.get('total_atoms_area_nm2', np.nan),
                    "Atoms_Density": atoms_analysis.get('atoms_density_in_contamination_nm2', np.nan),
                    "Atoms": json.dumps(atoms_analysis.get('atoms', [])),
                    "Circularities": json.dumps(contamination.get('shape_metrics', {}).get('circularity', [])),
                    "Roundness": json.dumps(contamination.get('shape_metrics', {}).get('roundness', [])),
                    "Feret_Diameter": json.dumps(contamination.get('shape_metrics', {}).get('feret_diameter', [])),
                    "Aspect_Ratio": json.dumps(contamination.get('shape_metrics', {}).get('aspect_ratio', []))
                })
                
        except Exception as e:
            print(f"Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()

        # 4. Add filter parameters to CSV data
        try:
            new_data['Threshold_Method'] = self.thresh_method_dropdown.value
            filter_params = {}
            for param, widget in [
                ("Analysis_Type", self.analysis_type_dropdown),
                ("Feature_Analysis_Type", self.feature_analysis_type_dropdown),
                ("Kernel_Size", self.kernel_size_slider),
                ("Threshold1", self.threshold_slider1),
                ("Threshold2", self.threshold_slider2),
                ("Threshold_SA1", self.threshold_sa_slider1),
                ("Threshold_SA2", self.threshold_sa_slider2),
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
                ("Opening2", self.opening2_slider),
                ("Closing2", self.closing2_slider),
                ("Dilation2", self.dilation2_slider),
                ("Erosion2", self.erosion2_slider),
                ("Gradient2", self.gradient2_slider),
                ("Boundary2", self.boundary2_slider),
                ("Circularity", self.min_circularity_slider),
                ("Isolation Distance", self.min_isolation_slider),
                ("Make Circular threshold", self.make_circular_thresh_slider),
                ("SA_Cluster_Definer", self.single_atom_clusters_definer_slider),
                ("Brightness", self.brightness_slider),
                ("Sigmoid Alpha", self.sigmoid_alpha_slider),
                ("Sigmoid Beta", self.sigmoid_beta_slider),
                ("Exp Transform", self.exp_transform_checkbox),
                ("Log Transform", self.log_transform_checkbox),
                ("Min_Clean_Cont_Area_nm2", self.min_clean_cont_area_slider),
                ("Max_Clean_Cont_Area_nm2", self.max_clean_cont_area_slider),
                ("Min_Cluster_Area_nm2", self.min_cluster_area_slider),
                ("Max_Cluster_Area_nm2", self.max_cluster_area_slider),
                ("Contour_retrieval_modes", self.contour_retrieval_dropdown),
                ("Contour_approximation_methods", self.contour_approximation_dropdown),
                ("Kmeans_Initialization", self.kmeans_initial_dropdown),
                ("Kmeans_Attempts_Number", self.kmeans_attempts),
                ("Kmeans_Clusters_Number", self.kmeans_clusters_number),
                ("Kmeans_Epsilon", self.kmeans_epsilon),
                ("Thresh_Method", self.thresh_method_dropdown),
                ("Otsu_Classes_Number", self.iterative_otsu_classes_number),
                ("Otsu_Region_Selection", self.iterative_otsu_region_selection)
            ]:
                threshold_method = self.thresh_method_dropdown.value
        
                if threshold_method == 'Manual':
                    # Set K-means parameters to NaN
                    filter_params.update({
                        "Kmeans_Initialization": np.nan,
                        "Kmeans_Attempts_Number": np.nan,
                        "Kmeans_Clusters_Number": np.nan,
                        "Kmeans_Epsilon": np.nan
                    })
                elif threshold_method == 'K-means':
                    # Set manual threshold parameters to NaN
                    filter_params.update({
                        "Threshold_SA1": np.nan,
                        "Threshold_SA2": np.nan
                    })
                elif threshold_method == 'Iterative Otsu':
                    # Set manual and K-means parameters to NaN
                    filter_params.update({
                        "Threshold_SA1": np.nan,
                        "Threshold_SA2": np.nan,
                        "Kmeans_Initialization": np.nan,
                        "Kmeans_Attempts_Number": np.nan,
                        "Kmeans_Clusters_Number": np.nan,
                        "Kmeans_Epsilon": np.nan
                    })
                    
                new_data.update(filter_params)
                print("Filter parameters collected")
                try:
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
                except AttributeError:
                    print(f"Warning: Widget for {param} not found")
                    filter_params[param] = np.nan

            new_data.update(filter_params)
            print("Filter parameters collected")
            
        except Exception as e:
            print(f"Parameter collection failed: {str(e)}")
            import traceback
            traceback.print_exc()

        # 5. Save to CSV
        try:
            print(f"Saving to: {os.path.abspath(filename)}")
            
            # Create new DataFrame row with enforced column order
            new_row = pd.DataFrame([new_data])
            
            # Reorder columns to match desired order
            new_row = new_row.reindex(columns=COLUMNS_ORDER + 
                                    [col for col in new_row.columns if col not in COLUMNS_ORDER])
            
            if os.path.isfile(filename):
                # Read existing data
                existing = pd.read_csv(filename)
                
                # Create complete column list (desired order + any extra columns)
                complete_columns = COLUMNS_ORDER + [col for col in existing.columns if col not in COLUMNS_ORDER]
                
                # Reorder existing columns
                existing = existing.reindex(columns=complete_columns)
                
                # Add any missing columns from new_data
                for col in new_data.keys():
                    if col not in existing.columns:
                        existing[col] = np.nan
                        
                # Reorder again after adding new columns
                existing = existing.reindex(columns=complete_columns + 
                                        [col for col in new_data.keys() if col not in complete_columns])
                
                # Find existing row for this slice
                mask = existing['Slice'] == slice_number
                if mask.any():
                    # Update existing row
                    idx = mask.idxmax()
                    for col in new_data:
                        if col in existing.columns:
                            existing.loc[idx, col] = new_row[col].values[0]
                    existing.to_csv(filename, index=False)
                    print(f"Updated existing row for slice {slice_number}")
                else:
                    # Append new row
                    updated = pd.concat([existing, new_row], ignore_index=True)
                    updated.to_csv(filename, index=False)
                    print(f"Appended new row for slice {slice_number}")
            else:
                # Create new file with enforced column order
                new_row.to_csv(filename, index=False)
                print(f"Created new file: {filename}")

            csv_path = os.path.abspath(filename)
            dir_path = os.path.dirname(csv_path)
            self._force_filesystem_refresh(dir_path)
            print(f"Successfully saved data to {csv_path}")
                
        except Exception as e:
            print(f"CSV save failed: {str(e)}")
            import traceback
            traceback.print_exc()


    
    def _interactive_image_analysis(self):
        return  interactive_output(self.analyse_features, {'slice_number': self.slice_slider, 'kernel': self.kernel_size_slider,
                                                            'threshold1': self.threshold_slider1, 'threshold2': self.threshold_slider2,
                                                            'threshold_sa1': self.threshold_sa_slider1,
                                                            'threshold_sa2': self.threshold_sa_slider2,
                                                            'gamma': self.gamma_slider, 'clahe_clip': self.clahe_clip_slider, 'clahe_tile': self.clahe_tile_slider,
                                                            'gaussian_sigma': self.gaussian_sigma_slider, 'contrast': self.contrast_slider,
                                                            'min_clean_cont_area': self.min_clean_cont_area_slider, 'max_clean_cont_area': self.max_clean_cont_area_slider, 
                                                            'min_cluster_area': self.min_cluster_area_slider, 'max_cluster_area': self.max_cluster_area_slider,
                                                            'double_gaussian_sigma1': self.double_gaussian_slider1, 'double_gaussian_sigma2': self.double_gaussian_slider2,
                                                            'double_gaussian_weight': self.double_gaussian_weight_slider, 
                                                            'iteration_opening': self.opening_slider, 
                                                            'iteration_erosion': self.erosion_slider,
                                                            'iteration_dilation': self.dilation_slider,
                                                            'iteration_closing': self.closing_slider,
                                                            'iteration_gradient': self.gradient_slider,
                                                            'iteration_boundary': self.boundary_slider,
                                                            'min_circularity': self.min_circularity_slider,
                                                            'isolation_distance': self.min_isolation_slider,
                                                            'opening2': self.opening2_slider, 
                                                            'closing2': self.closing2_slider,
                                                            'dilation2': self.dilation2_slider,
                                                            'erosion2': self.erosion2_slider,
                                                            'gradient2': self.gradient2_slider,
                                                            'boundary2': self.boundary2_slider,
                                                            'sa_cluster_definer': self.single_atom_clusters_definer_slider,
                                                            'make_circular': self.make_circular_thresh_slider,
                                                            'brightness': self.brightness_slider,
                                                            'sigmoid_alpha': self.sigmoid_alpha_slider,
                                                            'sigmoid_beta': self.sigmoid_beta_slider,
                                                            'exp_transform': self.exp_transform_checkbox,
                                                            'log_transform': self.log_transform_checkbox,
                                                            'contour_retrieval_modes': self.contour_retrieval_dropdown,
                                                            'contour_approximation_methods':self.contour_approximation_dropdown,
                                                            'x' : self.region_x_slider,
                                                            'y' : self.region_y_slider,
                                                            'w' : self.region_width_text,
                                                            'kmeans_init': self.kmeans_initial_dropdown,
                                                            'attempts_number': self.kmeans_attempts,
                                                            'k_number': self.kmeans_clusters_number,
                                                            'epsilon': self.kmeans_epsilon,
                                                            'otsu_classes_number': self.iterative_otsu_classes_number,
                                                            'otsu_region_selection': self.iterative_otsu_region_selection,
                                                            'colormap': self.colormap_dropdown, 'masked_color': self.mask_color_dropdown})




if __name__ == "__main__":

    font_path = "/home/somar/.fonts/SourceSansPro-Semibold.otf" 

    stacks = ['/home/somar/Desktop/2025/Data for publication/Multilayer graphene/Sample 2476/stacktest1.h5']
    stacks_ssb = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/SSB reconstruction of 4d STEM data/stack_ssbs.h5']
    stacks_ssb1 = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/SSB reconstruction of 4d STEM data/stack.h5']

    stacks_adf = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/ADF images/stack.h5']
    calibrated_images = FeaturesAnalysis(stacks_ssb1[0], font_path=font_path)
