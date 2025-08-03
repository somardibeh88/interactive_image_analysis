"""
Module for converting VCR images to GIF format
Author: Somar Dibeh
Date: 2025-07-31
"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from imageanalysis.calibrated_images_class import CalibratedImages  
from imageanalysis.fft_calibration_class import FFTCalibration
from data_loader import DataLoader
import matplotlib.pyplot as plt
import cv2
import imageio
from ipywidgets import (
    interactive_output, HBox, VBox, FloatSlider, IntSlider,
    Checkbox, Button, Output, Dropdown, IntText, FloatText,
    Text, HTML, Tab, Layout, GridBox, Label, ToggleButton
)
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output
from joint_widgets import create_widgets
from imageanalysis.calibrated_images_class import CalibratedImages


cv2.ocl.setUseOpenCL(False)


class VCRToGIF:
    def __init__(self, vcr_images_path, font_path=None):

        # Get the widgets for the GUI from the joint_widgets module
        dict_widgets = create_widgets()
        for key, value in dict_widgets.items():
            setattr(self, key, value)

        # Read the VCR images and their metadata using the DataLoader class
        self.vcr_images_path = vcr_images_path
        self.stack = CalibratedImages(vcr_images_path, display_widgets=False)
        self.images = self.stack.raw_data
        self.metadata = self.stack.raw_metadata
        self.font_path = font_path

        self.calibration_factor = None
        self.ref_fov = None
        self.calibration_display = Output()

        self.slice_slider = self.stack.slice_slider
        self.image_selector = self.stack.image_selector
        self.image_selector.observe(self._sync_sliders, names='value')
        self.slice_slider.observe(self._sync_sliders, names='value')

        self.fft_calibration =  FFTCalibration(self.stack, self.vcr_images_path, display_fft=False, 
                                              font_path=self.font_path, font_size=8)
        self.fft_calibration.calibration_controls.layout.display = 'none'


        self.play_pause_button = ToggleButton(
            value=False, description='Play/Pause', icon='play', layout=Layout(width='100px'))
        self.play_pause_button.observe(self.toggle_play_pause, names='value')

        self.create_processing_tab = self.stack.create_processing_tab()
        self.create_calibration_tab = self.stack.create_calibration_tab()
        self.create_save_tab = self.stack.create_save_tab()

        self.ref_image_index = self.image_selector.value
        self.update_reference_image({'new': self.ref_image_index}) 
        self.setup_observers = self.stack.setup_observers()



        # Create tabs
        self.tabs = Tab()
        self.tabs.children = [
            VBox([self.create_processing_tab], layout={'padding': '5px'}),
            VBox([self.create_calibration_tab], layout={'padding': '5px'}),
            VBox([self.create_save_tab])
            ]
        
        self.tabs.set_title(0, 'Processing')
        self.tabs.set_title(1, 'Calibration')
        self.tabs.set_title(2, 'Save')
        self.tabs.layout = Layout(width='100%')
        
        
        # Display the widgets and tabs
        display(VBox([self.tabs, self.display_save_vcr()], layout={ 'width': '1000px', 'padding': '5px'}))



    def update_reference_image(self, change):
        """Update reference image when selector changes"""
        self.ref_image_index = change['new']
        self.ref_image = self.stack.raw_data[self.ref_image_index]
        self.ref_image_shape = self.ref_image.shape[0]


    def _sync_sliders(self, change):
        """Keep image selector and slice slider in sync"""
        if change['owner'] == self.image_selector:
            self.slice_slider.value = change['new']
        else:
            self.image_selector.value = change['new']


    def toggle_play_pause(self, change):
        """
        Toggle the play/pause state of the animation.
        """
        if change['new']:
            self.play_pause_button.icon = 'pause'
            self.animate_vcr()
        else:
            self.play_pause_button.icon = 'play'
            if hasattr(self, 'ani'):
                self.ani.event_source.stop()


    def process_vcr(self, gamma, clahe_clip, clahe_tile, gaussian_sigma, contrast, double_gaussian_sigma1, double_gaussian_sigma2,
                                double_gaussian_weight, kernel, brightness, sigmoid_alpha, sigmoid_beta, colormap=None, slice_number=None):
        """
        Display the save VCR section with the current settings.
        """
        with self.calibration_display:
            clear_output(wait=True)
        
        if self.images is None or len(self.images) == 0:
            print("No images to display.")
            return
        img = self.images[slice_number] if slice_number is not None else self.images[0]
        nm_per_pixel, _ = self.fft_calibration.get_calibrated_image(img, slice_number=slice_number)

        vcr_images = self.stack.apply_filters(img, gamma, clahe_clip, clahe_tile, gaussian_sigma,
                                              contrast, double_gaussian_sigma1, double_gaussian_sigma2,
                                              double_gaussian_weight, kernel, brightness, sigmoid_alpha, sigmoid_beta)
        if vcr_images is None:
            print("No images to display after applying filters.")
            return


        
    def display_save_vcr(self):
        output = interactive_output(
            self.process_vcr,
            {
                'gamma': self.gamma_slider,
                'clahe_clip': self.clahe_clip_slider,
                'clahe_tile': self.clahe_tile_slider,
                'gaussian_sigma': self.gaussian_sigma_slider,
                'contrast': self.contrast_slider,
                'double_gaussian_sigma1': self.double_gaussian_slider1,
                'double_gaussian_sigma2': self.double_gaussian_slider2,
                'double_gaussian_weight': self.double_gaussian_weight_slider,
                'kernel': self.kernel_size_slider,
                'brightness': self.brightness_slider,
                'sigmoid_alpha': self.sigmoid_alpha_slider,
                'sigmoid_beta': self.sigmoid_beta_slider,
                'colormap': self.colormap_dropdown,
                'slice_number': self.slice_slider
            }
        )
        return output