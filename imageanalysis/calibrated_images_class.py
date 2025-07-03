from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from ipywidgets import (
    interactive_output, HBox, VBox, FloatSlider, IntSlider,
    Checkbox, Button, Output, Dropdown, IntText, FloatText,
    Text, HTML, Tab, Layout, GridBox, Accordion, Label
)
from IPython.display import display, clear_output
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.offsetbox import AnchoredText
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import json
import h5py
import os
import cv2
from datetime import datetime

from data_loader import DataLoader
from filters import *
from fft_calibration_class import FFTCalibration
from joint_widgets import create_widgets

plt.rcParams.update({'font.size': 10})


class CalibratedImages():

    INTERPOLATION_MAP = {'Bilinear': cv2.INTER_LINEAR,
                         'Bicubic': cv2.INTER_CUBIC,
                         'Lanczos': cv2.INTER_LANCZOS4,
                         'Nearest': cv2.INTER_NEAREST,
                         'Area': cv2.INTER_AREA,}

    def __init__(self, stack_path=None, font_path=None):
        # Initialize widgets from shared dict
        widgets_dict = create_widgets()
        for key, value in widgets_dict.items():
            setattr(self, key, value)

        self.font_path = font_path
        self.stack_path = stack_path
        self.stack = DataLoader(stack_path) if stack_path else []
        self.metadata = self.stack.raw_metadata if self.stack else {}
        self.calibration_factor = None
        self.ref_fov = None
        self.calibration_display = Output()

        # Create widgets
        self.slice_slider = IntSlider(min=0, max=len(self.stack.raw_data)-1 if self.stack else 0, 
                                     value=0, description='Slice', continuous_update=False)
        self.image_name = Text(value='image.png', description='Name:')
        self.save_image_button = Button(description="Save Image", tooltip="Save image with scalebar")
        self.save_for_figure_button = Button(description="Save Figure", tooltip="Save publication-quality image")
        self.image_selector = Dropdown(options=[(f'Image {i}', i) for i in range(len(self.stack.raw_data))] if self.stack else [], 
                                      value=0, description='Image:')
        self.ref_image_index = self.image_selector.value
        self.update_reference_image({'new': self.ref_image_index}) 

        # Setup FFT calibration (but don't display it yet)
        self.fft_calibration = FFTCalibration(self.stack, self.stack_path, display_fft=False)
        self.fft_calibration.calibration_controls.layout.display = 'none'  
        self.line_profile_widgets = self.add_line_profile_widgets()
        self.region_profile_widgets = self.add_region_profile_widgets()

        # Create tabs
        self.tabs = Tab()
        self.tabs.children = [
            VBox([self.create_processing_tab()], layout={'padding': '5px'}),
            VBox([ self.create_region_tab()], layout={'padding': '5px'}),
            VBox([self.create_calibration_tab()], layout={'padding': '5px'}),
            VBox([self.create_save_tab()])
            ]
        self.tabs.set_title(0, 'Processing')
        self.tabs.set_title(1, 'Region')
        self.tabs.set_title(2, 'Calibration')
        self.tabs.set_title(3, 'Save')

        display(VBox([self.tabs, self.display_save_images()],layout={ 'width': '1000px', 'padding': '5px'}))
        self.setup_observers()

    def create_processing_tab(self):
        filter_grid = GridBox(
            children=[
                self.contrast_checkbox, self.contrast_slider,
                self.gaussian_checkbox, self.gaussian_sigma_slider,
                self.double_gaussian_checkbox, VBox([
                    self.double_gaussian_slider1, 
                    self.double_gaussian_slider2,
                    self.double_gaussian_weight_slider
                ]),
                self.gamma_checkbox, self.gamma_slider,
                self.clahe_checkbox, VBox([
                    self.clahe_clip_slider, 
                    self.clahe_tile_slider
                ]),
                self.brightness_checkbox, self.brightness_slider,
                self.sigmoid_checkbox, VBox([
                    self.sigmoid_alpha_slider, 
                    self.sigmoid_beta_slider
                ]),
                self.opening_checkbox, self.opening_slider,
                self.kernel_size_checkbox, self.kernel_size_slider,
                self.resize_checkbox, VBox([
                    self.resize_factor_slider, 
                    self.resize_method_dropdown
                ]),
                HBox([self.log_transform_checkbox, self.exp_transform_checkbox]), HTML("")
            ],
            layout=Layout(
                grid_template_columns='repeat(2, 1fr)',
                grid_gap='4px 6px',
                align_items='stretch',
            )
        )

        return VBox([
            HTML("<h4>Image Enhancement</h4>"),
            filter_grid,
            HTML("<h4>Display Options</h4>"),
            HBox([
                VBox([Label("Slice:"), self.slice_slider]),
                VBox([Label("Colormap:"), self.colormap_dropdown])
            ])
        ])

    def create_region_tab(self):
        region_grid = GridBox(
            children=[
                self.specific_region_checkbox, HTML(""),
                Label("  "), self.region_x_slider,
                Label("  "), self.region_y_slider,
                Label("  "), self.region_width_text,
                Label("  "), self.region_height_text,
            ],
            layout=Layout(
                grid_template_columns='max-content 1fr',
                grid_gap='4px 6px',
                align_items='stretch',
            )
        )
        return VBox([
            HTML("<h4>Region Selection</h4>"),
            region_grid
        ])
    
    
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
        ])
    

    def create_save_tab(self):
        save_grid = GridBox(
            children=[
                Label("Name:"), self.image_name,
                Label("Format:"), self.image_format_dropdown,
                self.scalebar_length_checkbox, self.scalebar_length_slider,
                self.dpi_checkbox, self.dpi_slider,
                self.save_image_button, self.save_for_figure_button
            ],
            layout=Layout(
                grid_template_columns='max-content 1fr',
                grid_gap='4px 6px',
                align_items='center'
            )
        )
        return VBox([
            HTML("<h4>Save Options</h4>"),
            save_grid
        ])



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
            self.gamma_checkbox: [self.gamma_slider],
            self.clahe_checkbox: [self.clahe_clip_slider, self.clahe_tile_slider],
            self.brightness_checkbox: [self.brightness_slider],
            self.sigmoid_checkbox: [self.sigmoid_alpha_slider, self.sigmoid_beta_slider],
            self.opening_checkbox: [self.opening_slider],
            self.kernel_size_checkbox: [self.kernel_size_slider],
            self.resize_checkbox: [self.resize_factor_slider, self.resize_method_dropdown],
            self.scalebar_length_checkbox: [self.scalebar_length_slider],
            self.dpi_checkbox: [self.dpi_slider],
            self.specific_region_checkbox: [
                self.region_x_slider, 
                self.region_y_slider,
                self.region_width_text,
                self.region_height_text
            ],
        }

        # Set initial visibility
        for checkbox, widgets in controls_map.items():
            for widget in widgets:
                widget.layout.display = '' if checkbox.value else 'none'
                
        # Create observers
        for checkbox, widgets in controls_map.items():
            checkbox.observe(
                lambda change, w=widgets: self.toggle_visibility(change, w), 
                names='value'
            )


    def toggle_visibility(self, change, widgets):
        display_value = '' if change['new'] else 'none'
        for widget in widgets:
            widget.layout.display = display_value
        
        # Reset values when hiding
        if not change['new']:
            if widgets == [self.region_x_slider, self.region_y_slider,
                        self.region_width_text, self.region_height_text]:
                self.region_x_slider.value = 0
                self.region_y_slider.value = 0
                self.region_width_text.value = 100
                self.region_height_text.value = 100


    def update_reference_image(self, change):
        """Update reference image when selector changes"""
        self.ref_image_index = change['new']
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]


    def define_global_font_matplotlib(self):
        from matplotlib import font_manager
            # Register the font
        font_manager.fontManager.addfont(self.font_path)
        # Extract font name
        import os
        font_name = font_manager.FontProperties(fname=self.font_path).get_name()
        # Set globally
        plt.rcParams['font.family'] = font_name


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
    

    def resize_image(self, image):
        """Enhanced resizing with dynamic factor and method selection"""
        if self.resize_checkbox.value and self.resize_factor_slider.value != 1.0:
            image = image.copy()  # Create a copy to avoid modifying the original
            method = cv2.INTER_CUBIC if self.resize_method_dropdown.value == 'Bicubic' else cv2.INTER_LINEAR
            new_size = (int(image.shape[1] * self.resize_factor_slider.value),
                        int(image.shape[0] * self.resize_factor_slider.value))
            return cv2.resize(image, new_size, interpolation=method)
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
            HBox([self.save_region_profile_button, self.region_profile_name]),])
    

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
            'angle': self.line_angle.value}
        
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
            'h': self.region_profile_height.value}
        
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



    def display_calibrated_image_with_scalebar(self, gamma, clahe_clip, clahe_tile, gaussian_sigma,contrast, double_gaussian_sigma1, double_gaussian_sigma2,
                                         double_gaussian_weight, kernel, brightness, sigmoid_alpha,sigmoid_beta, scalebar_length, exp_transform, log_transform, 
                                         x,y,h,w, resize_factor,resize_method, opening, line_x, line_y, line_length, line_width, line_angle,
                                        region_profile_x, region_profile_y, region_profile_width, region_profile_height, colormap=None, slice_number=None):
        
        with self.calibration_display:
            clear_output(wait=True)
        
        # Process image
        image = self.stack.raw_data[slice_number]
        print("Minimum value:", np.min(image), "Maximum value:", np.max(image))
        nm_per_pixel, _ = self.fft_calibration.get_calibrated_image(image, slice_number)
     

        image = closing(image, opening, kernel)

        print(f"Image {slice_number} shape:", image.shape)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255 
        image = self.apply_filters(image, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2, double_gaussian_weight, kernel,
                      brightness, sigmoid_alpha, sigmoid_beta)
        image_1 = image.copy()
        if self.resize_checkbox.value:
            interpolation_method = self.INTERPOLATION_MAP[self.resize_method_dropdown.value]
            image = cv2.resize(image, (int(image.shape[1] * self.resize_factor_slider.value), int(image.shape[0] * self.resize_factor_slider.value)), interpolation=interpolation_method)

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

        # Clear previous patches
        for ax in self.axs.flatten():
            for patch in ax.patches[:]:
                patch.remove()



        self.axs[0,0].imshow(image, cmap=colormap)
        self.add_scalebar(self.axs[0,0], nm_per_pixel, scalebar_length)
        self.axs[0,0].set_title('Original Image')
                
        if self.specific_region_checkbox.value is True:
            x = self.region_x_slider.value
            y = self.region_y_slider.value
            w = self.region_width_text.value
            h = self.region_height_text.value
            rect = patches.Rectangle((x, y), w, h,
                                    linewidth=1, edgecolor='red', facecolor='none')
            self.axs[0,0].add_patch(rect)
            image_1 = image[y:y+h, x:x+w]
            self.axs[1,0].imshow(image_1, cmap=colormap)

        else:
            self.axs[1,0].imshow(image, cmap=colormap)
            self.add_scalebar(self.axs[1,0], nm_per_pixel, scalebar_length)
        # Add intensity profile elements
        line_profile = None
        region_profile = None

        # Draw line profile if parameters are set
        if self.specific_region_checkbox.value and any([self.region_x_slider.value > 0, self.region_y_slider.value > 0]):
            # Get values in image coordinates
            length = self.region_width_text.value
            width = self.region_height_text.value
            angle = self.line_angle.value

            # Calculate line endpoints
            angle_rad = np.deg2rad(angle)
            dx = length * np.cos(angle_rad)
            dy = length * np.sin(angle_rad)

            # Create polygon for width visualization
            perp_angle = angle_rad + np.pi/2
            dx_perp = width/2 * np.cos(perp_angle)
            dy_perp = width/2 * np.sin(perp_angle)

            x0 = self.region_x_slider.value 
            y0 = self.region_y_slider.value + width/2 * np.sin(perp_angle)

            
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
        if self.specific_region_checkbox.value and  any([self.region_x_slider.value > 0, self.region_y_slider.value > 0]):
            rx = self.region_x_slider.value
            ry = self.region_y_slider.value
            rw = self.region_width_text.value
            rh = self.region_height_text.value
            
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
            'sigmoid_alpha': self.sigmoid_alpha_slider.value,
            'sigmoid_beta': self.sigmoid_beta_slider.value,
            'exp_transform': self.exp_transform_checkbox.value,
            'log_transform': self.log_transform_checkbox.value,
            'contrast': self.contrast_slider.value,
            'x' : self.region_x_slider.value,
            'y' : self.region_y_slider.value,
            'h' : self.region_height_text.value,
            'w' : self.region_width_text.value,  
            'resize_factor': self.resize_factor_slider.value,
            'resize_method': self.resize_method_dropdown.value,  
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
        if not self.scalebar_length_checkbox.value:
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
    
    def display_save_images(self):
        output =interactive_output( self.display_calibrated_image_with_scalebar,{'kernel': self.kernel_size_slider,
                                                                                 'gamma': self.gamma_slider,
                                                                                 'slice_number': self.slice_slider,
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
                                                                                'sigmoid_alpha': self.sigmoid_alpha_slider,
                                                                                'sigmoid_beta': self.sigmoid_beta_slider,
                                                                                'exp_transform': self.exp_transform_checkbox,
                                                                                'log_transform': self.log_transform_checkbox,
                                                                                'scalebar_length': self.scalebar_length_slider,
                                                                                'colormap': self.colormap_dropdown,
                                                                                'x' : self.region_x_slider,
                                                                                'y' : self.region_y_slider,
                                                                                'h' : self.region_height_text,
                                                                                'w' : self.region_width_text,
                                                                                'resize_factor': self.resize_factor_slider,
                                                                                'resize_method': self.resize_method_dropdown,
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
                                                                                'region_profile_height': self.region_profile_height})
        return output
                                                      
