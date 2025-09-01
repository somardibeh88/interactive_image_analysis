
"""""""""""""""""
Module for displaying and calibrating images using Fourier space calibration
Author: Somar Dibeh
"""""""""""""""""

from ipywidgets import (interactive_output, HBox, VBox, FloatSlider, IntSlider, 
                        Button, Output, Dropdown, IntText, FloatText, 
                        Text, HTML, Tab, Layout, GridBox)
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
import numpy as np
from . import fourier_scale_calibration as fsc
from .fourier_scale_calibration import *
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as patches
import json
import h5py
import os
import cv2
from .data_loader import DataLoader
from .filters import *
from .joint_widgets import *

plt.rcParams.update({'font.size': 8})


class FFTCalibration():
    materials = {'hBN': 2.504, 'Graphene': 2.46, 'MoS2': 3.212 }
    calibration_types = ['hexagonal','2nd-order-hexagonal', 'graphene-2L','graphene-3L', 'graphene-4L', 'graphene-5L']
    fft_orders = ['1st-order', '2nd-order']

    def __init__(self, stack, stack_path=None, display_fft=False, font_path=None, font_size=8):
               
        widgets_dict = create_widgets()
        for name, widget in widgets_dict.items():
            setattr(self, f"{name}_fft_calib", widget)
        self.font_path = font_path 
        self.font_size = font_size
        self.stack = stack
        self.metadata = self.stack.raw_metadata if hasattr(self.stack, 'raw_metadata') else None
        self.stack_path = stack_path if stack_path else getattr(stack, 'path', None)

        self.ref_fov = None
        self.calibrator = None
        self._last_calibration = None
        self._cached_nm_per_pixel = None
        self.calibration_factor = None
        self.display_fft = display_fft


        # For calibrating the images
        self.image_selector = Dropdown(options=[(f'Image {i}', i) for i in range(len(self.stack.raw_data))], 
                                      value=0, description='Image:', layout={'width': '95%'})
        self.image_selector.observe(self.update_reference_image, names='value')

        # # Initialize reference image from selector
        self.ref_image_index = self.image_selector.value
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]

        self.update_reference_image({'new': self.ref_image_index}) 

        # Add FOV editing widgets
        self.fov_input = FloatText(value=16.0, description='FOV (nm):', step=0.1, 
                                 style={'description_width': '90px'}, 
                                 layout={'width': '95%'})
        self.save_fov_button = Button(description="Save FOV", 
                                    tooltip="Save FOV to metadata",
                                    layout={'width': '95%'})



        # Save FFT image button
        self.save_fft_button = Button(description='Save FFT', layout={'width': '95%'})
        self.save_fft_button.on_click(self.save_fft_image)

        self.fft_image_name = Text(value='fft_calibration.svg', description='Name:', 
                                  style={'description_width': '60px'}, 
                                  layout={'width': '95%'})

        max_layers = 5
        self.layer_angles = [FloatSlider(min=0,max=360,value=0,step=0.25,
                                        description=f'L{i} Angle (°)',
                                        style={'description_width': '100px'},
                                        layout={'width': '95%', 'display': 'none'}) 
                            for i in range(2, max_layers+1)]
        self.fft_order_dropdown = Dropdown(options=self.fft_orders, value='1st-order',
                                         description='FFT Order:', layout={'width': '95%'})
        
        # For plotting and controlling the fft widgets
        calibration_widgets = [ self.image_selector, self.calibration_type_dropdown_fft_calib, self.materials_dropdown_fft_calib, 
                              self.min_sampling_slider_fft_calib, self.max_sampling_slider_fft_calib, self.gamma_fft_slider_fft_calib, 
                              self.gaussian_sigma_fft_slider_fft_calib, self.region_x_slider_fft_calib, self.region_y_slider_fft_calib, self.region_width_text_fft_calib, 
                              self.region_height_text_fft_calib, self.calibrate_region_checkbox_fft_calib, self.n_widget_slider_fft_calib, 
                              self.gamma_slider_fft_calib, self.gaussian_sigma_slider_fft_calib, self.contrast_slider_fft_calib, 
                              self.double_gaussian_slider1_fft_calib, self.double_gaussian_slider2_fft_calib, 
                              self.double_gaussian_weight_slider_fft_calib, self.kernel_size_slider_fft_calib, 
                              self.colormap_dropdown_fft_calib, self.fft_spots_rotation_slider_fft_calib, self.rolloff_slider_fft_calib, 
                              self.cuttoff_slider_fft_calib, *self.layer_angles, 
                              self.save_for_figure_checkbox_fft_calib, self.fft_order_dropdown]
        for widget in calibration_widgets:
            widget.observe(self.fft_calibrate, names='value')
        self.calibrate_button = Button(description='Calibrate', layout={'width': '95%'})
        self.calibrate_button.on_click(self.fft_calibrate)
        self.calibration_output = Output()
        self.calibration_display = Output()

        # Create compact tab layout
        # Image Processing Tab
        filter_grid = GridBox(
            children=[
                self.gaussian_checkbox_fft_calib, self.gaussian_sigma_slider_fft_calib,
                self.double_gaussian_checkbox_fft_calib, VBox([self.double_gaussian_slider1_fft_calib, 
                                                     self.double_gaussian_slider2_fft_calib,
                                                     self.double_gaussian_weight_slider_fft_calib]),
                self.gamma_checkbox_fft_calib, self.gamma_slider_fft_calib,
                self.contrast_checkbox_fft_calib, self.contrast_slider_fft_calib,
                self.colormap_dropdown_fft_calib
            ],
            layout=Layout(
                grid_template_columns='repeat(2, 1fr)',
                grid_gap='5px 10px',
                align_items='stretch'
            )
        )
        
        # Calibration Parameters Tab
        cal_grid = GridBox(
            children=[
                self.materials_dropdown_fft_calib, self.calibration_type_dropdown_fft_calib,
                self.fft_order_dropdown, self.image_selector,
                self.min_sampling_slider_fft_calib, self.max_sampling_slider_fft_calib,
                self.n_widget_slider_fft_calib, self.fft_spots_rotation_slider_fft_calib,
                self.rolloff_slider_fft_calib, self.cuttoff_slider_fft_calib,
                self.gamma_fft_slider_fft_calib, self.gaussian_sigma_fft_slider_fft_calib,
                VBox(self.layer_angles)
            ],
            layout=Layout(
                grid_template_columns='repeat(2, 1fr)',
                grid_gap='5px 10px',
                align_items='stretch'
            )
        )

        
        self.calibrate_region_checkbox_fft_calib.layout = Layout(grid_column='span 2', width='100%')
        # Region Selection Tab
        region_grid = GridBox(
            children=[
                # First row: Checkbox spanning both columns
                self.calibrate_region_checkbox_fft_calib,
                
                HTML(" ", layout={'grid_column': 'span 2'}), 
                
                # Second row: Position header
                HTML("<b>Position (px)</b>", layout={'grid_column': 'span 2'}),
                
                # Third row: X and Y controls
                HTML("  "), 
                self.region_x_slider_fft_calib,
                HTML("  "), 
                self.region_y_slider_fft_calib,
                HTML("  "), 
                # Fourth row: Size header
                HTML("<b>Size (px)</b>", layout={'grid_column': 'span 2'}),
                
                # Fifth row: Width and Height controls
                HTML("  "), 
                self.region_width_text_fft_calib,
                HTML("  "), 
                self.region_height_text_fft_calib
            ],
            layout=Layout(
                grid_template_columns='max-content 1fr',  # First column for labels, second for controls
                grid_template_rows='auto auto auto auto auto',  # Five rows
                grid_gap='5px 10px',
                align_items='center'
            )
        )
        
        # Save Options Tab
        save_grid = GridBox(
            children=[
                self.save_fft_button, self.fft_image_name,
                self.save_fov_button, self.fov_input,
                self.save_for_figure_checkbox_fft_calib
            ],
            layout=Layout(
                grid_template_columns='repeat(2, 1fr)',
                grid_gap='5px 10px',
                align_items='center'
            )
        )
        
        # Create tabs
        self.tabs = Tab()
        self.tabs.children = [
            VBox([filter_grid], layout={'padding': '5px'}),
            VBox([cal_grid], layout={'padding': '5px'}),
            VBox([region_grid], layout={'padding': '5px'}),
            VBox([save_grid], layout={'padding': '5px'})
        ]
        self.tabs.set_title(0, 'Filters')
        self.tabs.set_title(1, 'Calibration')
        self.tabs.set_title(2, 'Region')
        self.tabs.set_title(3, 'Save')
        
        # Main calibration controls
        self.calibration_controls = VBox([
            self.tabs,
            HBox([self.calibrate_button, self.apply_calibration_checkbox_fft_calib], 
                 layout={'margin': '10px 0', 'width': '100%'}),
            self.calibration_output,
            self.calibration_display
        ], layout={'display': 'none', 'width': '840px', 'padding': '5px'})

        calibration_checkboxes = [self.calibration_checkbox_fft_calib, self.gaussian_checkbox_fft_calib, self.double_gaussian_checkbox_fft_calib,
                                  self.calibrate_region_checkbox_fft_calib, self.contrast_checkbox_fft_calib, self.gamma_checkbox_fft_calib]
        for checkbox in calibration_checkboxes:
            checkbox.observe(self.toggle_calibration_controls, names='value')

        self.calibration_checkbox_fft_calib.observe(self.handle_calibration_checkbox, names='value')
        self.calibration_type_dropdown_fft_calib.observe(self.update_layer_controls, names='value')
        self.fft_order_dropdown.observe(self.update_fft_order, names='value')
        
        # Apply consistent styling to all sliders
        slider_style = {'description_width': '120px', 'handle_color': '#4285F4'}
        for slider in [self.min_sampling_slider_fft_calib, self.max_sampling_slider_fft_calib, 
                       self.n_widget_slider_fft_calib, self.fft_spots_rotation_slider_fft_calib,
                       self.rolloff_slider_fft_calib, self.cuttoff_slider_fft_calib, self.gamma_fft_slider_fft_calib,
                       self.gaussian_sigma_fft_slider_fft_calib, self.gamma_slider_fft_calib,
                       self.gaussian_sigma_slider_fft_calib, self.contrast_slider_fft_calib,
                       self.double_gaussian_slider1_fft_calib, self.double_gaussian_slider2_fft_calib,
                       self.double_gaussian_weight_slider_fft_calib]:
            slider.style = slider_style

        # if self.display_fft is True:
        #     display(self.calibration_checkbox_fft_calib, self.calibration_controls)
        # else:
        #     pass


    def toggle_calibration_controls(self, change):
        checkbox_slider_map = {
            self.calibration_checkbox_fft_calib: [self.calibration_controls],
            self.gaussian_checkbox_fft_calib: [self.gaussian_sigma_slider_fft_calib],
            self.double_gaussian_checkbox_fft_calib: [self.double_gaussian_slider1_fft_calib, 
                                           self.double_gaussian_slider2_fft_calib, 
                                           self.double_gaussian_weight_slider_fft_calib],
            self.contrast_checkbox_fft_calib: [self.contrast_slider_fft_calib],
            self.gamma_checkbox_fft_calib: [self.gamma_slider_fft_calib],
            self.calibrate_region_checkbox_fft_calib: [self.region_x_slider_fft_calib, self.region_y_slider_fft_calib, 
                                            self.region_width_text_fft_calib, self.region_height_text_fft_calib]
        }
        # Find which checkbox changed and update its corresponding sliders
        for checkbox, sliders in checkbox_slider_map.items():
            if change['owner'] == checkbox:
                visibility = '' if change['new'] else 'none'
                for slider in sliders:
                    slider.layout.display = visibility
                break


    def update_fft_order(self, change):
        fft_order = self.fft_order_dropdown.value


    def update_reference_image(self, change):
        """Update reference image when selector changes"""
        self.ref_image_index = change['new']
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]
        self._get_ref_fov()

    def update_colormap(self, change):
        """Handles colormap changes without triggering a full update."""
        self.update(colormap=self.colormap_dropdown_fft_calib.value)


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



    def update_layer_controls(self, change):
            """Update visible layer angle sliders based on selected layers"""
            calib_type = self.calibration_type_dropdown_fft_calib.value
            
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


    # Apply different filters
    def apply_filters_fft(self, image, gamma, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel):
        if self.gaussian_checkbox_fft_calib.value:
            image = apply_gaussian_blur(image, kernel, gaussian_sigma)

        # Apply contrast enhancement
        if self.contrast_checkbox_fft_calib.value:
            image = improve_contrast(image, contrast)  #Global contrast correction across the image

        # Apply Double Gaussian Filter
        if self.double_gaussian_checkbox_fft_calib.value:
            image = double_gaussian(image, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight)

        #Apply gammaa correction
        if self.gamma_checkbox_fft_calib.value:
            image = apply_gamma_correction(image, gamma)
        return image
    


    def apply_filters_fft_spots(self, image, gamma, gaussian_sigma):
        from scipy.ndimage import gaussian_filter
        if self.gaussian_checkbox_fft_calib.value:
            image = gaussian_filter(image, sigma=gaussian_sigma)
        if self.gamma_checkbox_fft_calib.value:
            image = apply_gamma_correction(image, gamma)
        return image



    def _get_ref_fov(self):
        if self.metadata is not None:
            # metadata = self.metadata[f"metadata_{self.ref_image_index:04d}"]
            metadata = self.metadata[self.ref_image_index]
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
    


    def get_calibrated_image(self, image, slice_number):

        if self._last_calibration == (slice_number, self.calibration_factor):
            return self._cached_nm_per_pixel, self._cached_nm_per_pixel**2
        # Calibration
        # meta = self.metadata[f"metadata_{slice_number:04d}"]
        meta = self.metadata[slice_number]
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
            
        if (self.calibration_factor and hasattr(self, 'ref_fov') and self.apply_calibration_checkbox_fft_calib.value and self.ref_fov is not None):
            scale_factor = (self.ref_image_shape / image.shape[0]) * (fov / self.ref_fov)
            nm_per_pixel = self.calibration_factor * scale_factor
            print(scale_factor, nm_per_pixel)
            nm2_per_pixel2 = nm_per_pixel ** 2
            fov_calibrated = image.shape[0] * nm_per_pixel
            print("Original image shape", image.shape[0],  "Reference image shape", self.ref_image_shape)
            print(f'Calibrated FOV: {fov_calibrated:.2f} nm', 'FOV from metadata:', fov)

        else:
            print('Not calibrated yet, using a default values of 0.01 nm/pixel')
            nm_per_pixel = 0.01
            nm2_per_pixel2 = 0.01

        self._last_calibration = (slice_number, self.calibration_factor)
        self._cached_nm_per_pixel = nm_per_pixel
        return nm_per_pixel, nm2_per_pixel2
    
    

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

    


    def fft_calibrate(self, b=None):
        """Updates calibration display when parameters change"""
        if not self.calibration_checkbox_fft_calib.value:
            return
        
        try:
            # Get reference image and parameters
            ref_image = self.ref_image.copy()
            gamma = self.gamma_slider_fft_calib.value
            gaussian_sigma = self.gaussian_sigma_slider_fft_calib.value
            contrast = self.contrast_slider_fft_calib.value
            double_gaussian_sigma1 = self.double_gaussian_slider1_fft_calib.value
            double_gaussian_sigma2 = self.double_gaussian_slider2_fft_calib.value
            double_gaussian_weight = self.double_gaussian_weight_slider_fft_calib.value
            kernel = self.kernel_size_slider_fft_calib.value
            colormap = self.colormap_dropdown_fft_calib.value
            fft_order = int(self.fft_order_dropdown.value[0])
            print(f"FFT Order: {fft_order}")
            ref_image = self.apply_filters_fft(ref_image, gamma, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel)
            ref_image_plotiing = ref_image.copy()
            # Apply region selection if enabled
            if self.calibrate_region_checkbox_fft_calib.value:
                x = self.region_x_slider_fft_calib.value
                y = self.region_y_slider_fft_calib.value
                w = self.region_width_text_fft_calib.value
                h = self.region_height_text_fft_calib.value
                ref_image = ref_image[y:y+h, x:x+w]
            layer_angles=[slider.value for slider in self.layer_angles if slider.layout.display != 'none']
            # Create calibrator with current parameters
            self.calibrator = fsc.FourierSpaceCalibrator(
                template=self.calibration_type_dropdown_fft_calib.value,
                lattice_constant=self.materials[self.materials_dropdown_fft_calib.value],
                max_sampling=self.max_sampling_slider_fft_calib.value,
                min_sampling=self.min_sampling_slider_fft_calib.value,
                normalize_azimuthal=False,
                layer_angles=layer_angles,
                fft_order=fft_order,)
            
            # Window FFT parameters
            rolloff = self.rolloff_slider_fft_calib.value
            cutoff = self.cuttoff_slider_fft_calib.value

            # Perform calibration
            self.calibration_factor = self.calibrator.calibrate(ref_image) / 10

            # Update display
            with self.calibration_display:
                clear_output(wait=True)
                if not hasattr(self, 'cal_fig') or not plt.fignum_exists(self.cal_fig.number):
                    self.cal_fig, self.cal_ax1 = plt.subplots(1, 2, figsize=(6, 3))
                    self.cal_fig.show()
                else:
                    # Clear previous content
                    for ax in self.cal_ax1.flatten():
                        ax.cla()
                    
                # Show selected region
                self.current_ref_image_fft = ref_image
                self.cal_ax1[0].imshow(ref_image_plotiing, cmap=colormap)
                if self.calibrate_region_checkbox_fft_calib.value:
                    rect = patches.Rectangle((x, y), w, h, 
                                        linewidth=2, edgecolor='red', facecolor='none')
                    self.cal_ax1[0].add_patch(rect)
                self.cal_ax1[0].set_title(f'Calibration Region (Image {self.ref_image_index})')
                gamma_fft = self.gamma_fft_slider_fft_calib.value
                gaussian_sigma_fft = self.gaussian_sigma_fft_slider_fft_calib.value
                # Show FFT analysis
                ft_image = np.fft.fftshift(fsc.windowed_fft(ref_image, cf=cutoff, rf=rolloff))
                ft_vis = np.log(np.abs(ft_image) + 1e-6)
                ft_vis = self.apply_filters_fft_spots(ft_vis, gamma_fft, gaussian_sigma_fft)
                spots = self.calibrator.get_spots()
                spots = fsc.rotate(spots, self.fft_spots_rotation_slider_fft_calib.value, center=(ft_vis.shape[0] // 2, ft_vis.shape[1] // 2))
                n = self.n_widget_slider_fft_calib.value
                # Current FFT data for saving
                self.current_spots = spots
                self.current_n = self.n_widget_slider_fft_calib.value
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
                print(self.ref_image.shape, self.ref_image_index, self.ref_image.dtype)
                print(f'Calibration factor: {self.calibration_factor:.6f} nm/pixel')
                print(f'Reference FOV: {self.ref_fov} nm')
                print(f'Calibrated FOV: {self.ref_image_shape * self.calibration_factor:.3f} nm')
                print('Apply calibration using checkbox below')

        except Exception as e:
            with self.calibration_output:
                print(f'Calibration failed: {str(e)}')


    def save_fft_image(self, b):
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        # Setup font properties
        font_prop = fm.FontProperties(fname=self.font_path, size=self.font_size, weight='semibold') 

        # Ensure SVG text remains as vector elements
        mpl.rcParams['svg.fonttype'] = 'none'

        """Saves both the calibration image and FFT analysis as separate SVG files"""
        if not hasattr(self, 'current_ft_vis') or not hasattr(self, 'current_ref_image_fft'):
            with self.calibration_output:
                print("No images to save. Please run calibration first")
            return

        base_name = self.fft_image_name.value.replace('.svg', '')
        ref_filename = f"{base_name}_slice({self.ref_image_index})_ref.svg"
        fft_filename = f"{base_name}_slice({self.ref_image_index})_fft.svg"
        if '/' in ref_filename:
            dir_name, ref_img_fft = os.path.split(ref_filename)
            if dir_name == '':
                dir_name = '.'         # This will avoid having error if the path is just a filename and no directory included
            os.makedirs(dir_name, exist_ok=True)
            ref_filename = os.path.join(dir_name, ref_img_fft)
        if '/' in fft_filename:
            dir_name, fft_img_fft = os.path.split(fft_filename)
            if dir_name == '':
                dir_name = '.'
            # Create directory if it doesn't exist
            os.makedirs(dir_name, exist_ok=True)
            fft_filename = os.path.join(dir_name, fft_img_fft)


        fig_ref, ax_ref = plt.subplots(figsize=(8, 4), dpi=350)
        im = ax_ref.imshow(self.current_ref_image_fft, cmap=self.current_colormap)
        im.set_rasterized(True)

        # Draw region rectangle if region selection is active
        if self.calibrate_region_checkbox_fft_calib.value:
            x = self.region_x_slider_fft_calib.value
            y = self.region_y_slider_fft_calib.value
            w = self.region_width_text_fft_calib.value
            h = self.region_height_text_fft_calib.value
            rect = patches.Rectangle((x, y), w, h, 
                                linewidth=1, edgecolor='red', facecolor='none')
            ax_ref.add_patch(rect)

        # Add informational box
        if not self.save_for_figure_checkbox_fft_calib.value:
            info_text = (f"Slice: {self.ref_image_index}\ncalibration factor: {self.calibration_factor:.6f} nm/pixel\n"
                        f"FOV: {self.ref_fov:.2f} nm\nCalibrated FOV: {self.ref_image_shape * self.calibration_factor:.2f} nm")
            anchored_text = AnchoredText(info_text, loc='upper left',
                                        frameon=True, pad=0.5, borderpad=0.5)
            anchored_text.txt._text.set_fontproperties(font_prop)
            anchored_text.patch.set_boxstyle("round,pad=0.3")
            anchored_text.patch.set_facecolor("white")
            anchored_text.patch.set_alpha(0.9)
            ax_ref.add_artist(anchored_text)

            ax_ref.axis('off')
            plt.savefig(ref_filename, bbox_inches='tight', format='svg', pad_inches=0)
            plt.close(fig_ref)

            # Save FFT analysis
            fig_fft, ax_fft = plt.subplots(figsize=(8, 4), dpi=350)
            im1 = ax_fft.imshow(self.current_ft_vis, cmap=self.current_colormap)
            im1.set_rasterized(True)
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
            txt_angle_mismatches = f"Angle mismatches: {angle_text}" if visible_angles and self.calibration_type_dropdown_fft_calib.value not in ['hexagonal', '2nd-order-hexagonal'] else " "
            info_text = (
                f"FFT calibration of the image (slice {self.ref_image_index})\n"
                f"Calibration factor: {self.calibration_factor:.5f} nm/pixel\n"
                f"{txt_angle_mismatches}")

            anchored_text = AnchoredText(info_text, loc='upper left',
                                        frameon=True, pad=0.5, borderpad=0.5)
            anchored_text.txt._text.set_fontproperties(font_prop)
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
            txt_angle_mismatches = f"Angle mismatches: {angle_text}" if visible_angles and self.calibration_type_dropdown_fft_calib.value not in ['hexagonal', '2nd-order-hexagonal'] else " "
            # Compose info box content
            if self.calibration_type_dropdown_fft_calib.value not in ['hexagonal', '2nd-order-hexagonal']:
                info_text = (f"Number of layers: {self.calibration_type_dropdown_fft_calib.value[-2]}\n"
                    f"{txt_angle_mismatches}")
                # Save the images with the same name
                fig_ref, ax_ref = plt.subplots(figsize=(8, 4), dpi=350)
                anchored_text = AnchoredText(info_text, loc='upper left',
                                frameon=True, pad=0.5, borderpad=0.5)
                anchored_text.txt._text.set_fontproperties(font_prop)
                anchored_text.patch.set_boxstyle("round,pad=0.3")
                anchored_text.patch.set_facecolor("white")
                anchored_text.patch.set_alpha(0.9)
                ax_ref.add_artist(anchored_text)

            else:
                fig_ref, ax_ref = plt.subplots(figsize=(8, 4), dpi=350)
            im = ax_ref.imshow(self.current_ref_image_fft, cmap=self.current_colormap)
            im.set_rasterized(True)
            ax_ref.axis('off')
            plt.savefig(ref_filename, bbox_inches='tight', format='svg', pad_inches=0)
            plt.close(fig_ref)

            # Save FFT analysis
            fig_fft, ax_fft = plt.subplots(figsize=(8, 4), dpi=350)
            ax_fft.imshow(self.current_ft_vis, cmap=self.current_colormap)

            if self.calibration_type_dropdown_fft_calib.value not in ['hexagonal', '2nd-order-hexagonal']:
                anchored_text1 = AnchoredText(info_text, loc='upper left',
                    frameon=True, pad=0.5, borderpad=0.5)
                anchored_text.txt._text.set_fontproperties(font_prop)
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


if __name__ == "__main__":
    # Example usage
    stacks_ssb1 = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/SSB reconstruction of 4d STEM data/stack.h5']
    from imageanalysis.data_loader import DataLoader
    img = DataLoader(stacks_ssb1[0])
    fft_calibration = FFTCalibration(img)
    fft_calibration.fft_calibrate()

