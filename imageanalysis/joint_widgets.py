""""""""""""""""""""""'""""""
Joint widgets for interactive image analysis.

Author: Somar Dibeh
"""""""""""""""""""""""""""""

import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from ipywidgets import (interactive_output, HBox, VBox, FloatSlider, IntSlider, 
                        Checkbox, Button, Output, Dropdown, IntText, FloatText, 
                        Text, HTML, Tab, Accordion, Layout, GridBox, ToggleButton)

from matplotlib import colormaps

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

###################### Droplists  ######################
materials = {'hBN': 2.504, 'Graphene': 2.46, 'MoS2': 3.212 }

calibration_types = ['hexagonal','2nd-order-hexagonal', 'graphene-2L','graphene-3L', 'graphene-4L', 'graphene-5L']

contour_retrieval_modes = { 'RETR_EXTERNAL': cv2.RETR_EXTERNAL,
                            'RETR_CCOMP': cv2.RETR_CCOMP}

contour_approximation_methods = { 'CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE,
                                    'CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE,
                                    'CHAIN_APPROX_TC89_L1': cv2.CHAIN_APPROX_TC89_L1,
                                    'CHAIN_APPROX_TC89_KCOS': cv2.CHAIN_APPROX_TC89_KCOS}



fft_orders = ['1st-order', '2nd-order']

analysis_type = ['Clean_area_analysis', 'Contaminated_area_analysis']
feature_analysis_type = ['Single_atom_clusters_analysis', 'Defects_analysis'] 



def load_moreland_colormap(filepath, name):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    positions = data[:, 0]
    colors = data[:, 1:4]
    return LinearSegmentedColormap.from_list(name, list(zip(positions, colors)), N=256)

BASE_PATH = os.path.join(os.path.dirname(__file__), "colormaps")
print(BASE_PATH)

files = {
    "blackbody": "black-body-table-float-0256.csv",
    "kindlmann": "kindlmann-table-float-0256.csv",
}


custom_colormaps = {
    "blackbody": load_moreland_colormap(os.path.join(BASE_PATH, "black-body-table-float-0256.csv"), "blackbody"),
    "kindlmann": load_moreland_colormap(os.path.join(BASE_PATH, "kindlmann-table-float-0256.csv"), "kindlmann"),
}

# Register only your custom colormaps
for name, cmap in custom_colormaps.items():
    if name not in colormaps:
        colormaps.register(cmap)


def create_widgets():
    """
    Create and return a dictionary of widgets used in the image analysis application.
    This function initializes various widgets such as sliders, dropdowns, checkboxes, etc.
    for user interaction in the image analysis process.
        """
    contour_retrieval_modes = { 'RETR_EXTERNAL': cv2.RETR_EXTERNAL,
                                'RETR_CCOMP': cv2.RETR_CCOMP}

    contour_approximation_methods = { 'CHAIN_APPROX_NONE': cv2.CHAIN_APPROX_NONE,
                                        'CHAIN_APPROX_SIMPLE': cv2.CHAIN_APPROX_SIMPLE,
                                        'CHAIN_APPROX_TC89_L1': cv2.CHAIN_APPROX_TC89_L1,
                                        'CHAIN_APPROX_TC89_KCOS': cv2.CHAIN_APPROX_TC89_KCOS}

    kmeans_initialization_methods = {'K-means++': cv2.KMEANS_PP_CENTERS, 'Random': cv2.KMEANS_RANDOM_CENTERS}
    ###################### Dropdowns ######################
    colormap_dropdown = Dropdown(
        options=[
                'gray', 'viridis', 'cividis', 'plasma', 'magma', 'inferno',
                # Blue–Green
                'GnBu', 'BuGn', 'PuBuGn', 'YlGnBu', 'BuPu',
                # Blue–Red
                'bwr', 'seismic', 'coolwarm', 'RdBu',
                # Purple–Green
                'PRGn', 'PiYG', 'PuOr',
                # Brown–Blue
                'BrBG',
                # Heat-like
                'hot', 'afmhot', 'gist_heat',
                # Special
                'Spectral', 'twilight',
                'blackbody', 'kindlmann', 
            ],
        value='gray', description='Colormap:', continuous_update=False,
        layout={'width': '95%'})

    # Thresholding settings : for defining contour retrieval modes and approximation methods
    contour_retrieval_dropdown = Dropdown( options=list(contour_retrieval_modes.keys()),
                                                        value='RETR_EXTERNAL',
                                                        description='Retrieval Mode:',
                                                        style={'description_width': '140px'},
                                                        layout={'width': '360px'})

    contour_approximation_dropdown = Dropdown( options=list(contour_approximation_methods.keys()),
                                                        value='CHAIN_APPROX_TC89_KCOS',
                                                        description='Approximation:',
                                                        style={'description_width': '140px'},
                                                        layout={'width': '360px'})

    materials_dropdown = Dropdown(options=list(materials.keys()), value='Graphene', description='Material:')

    calibration_type_dropdown = Dropdown(options=calibration_types, value='hexagonal', description='Calibration Type:')

    fft_order_dropdown = Dropdown(options=fft_orders, value='1st-order',description='FFT Order:')

    image_format_dropdown = Dropdown(options=['png', 'svg'], value='png', description='Image format:', style={'description_width': '160px'}, layout={'width': '300px'})

    mask_color_dropdown = Dropdown(options=['red', 'white','black', 'green', 'blue'], value='red', description='Mask Color',continuous_update=False)

    resize_method_dropdown = Dropdown(options=['Bilinear', 'Bicubic', 'Lanczos', 'Nearest', 'Area'], value='Bicubic', description='Method:', style={'description_width': '80px'}, layout={'width': '95%'})

    analysis_type_dropdown = Dropdown(options=analysis_type, value='Clean_area_analysis', description='Analysis Type:', style={'description_width': '160px'}, layout={'width': '300px'})
    feature_analysis_type_dropdown = Dropdown(options=feature_analysis_type, value='Single_atom_clusters_analysis', description='Feature Analysis Type:', style={'description_width': '260px'}, layout={'width': '500px'})


    image_name = Text(value='image', description='Name:', style={'description_width': '160px'}, layout={'width': '95%'})
    save_image_button = Button(description="Save Image", tooltip="Save image with scalebar", layout={'width': '220px'})
    save_for_figure_button = Button(description="Save Figure", tooltip="Save image with scalebar", layout={'width': '160px'})
    save_button = Button(description="Save", tooltip="Save the data point to a CSV file. A new file will be created if one does not exist", layout={'width': '220px'})
    filename_input = Text(value='analysis_results', description='Filename:', style={'description_width': '140px'}, layout={'width': '96%'})

    ###################### VCR widget ######################
    display_vcr_checkbox = Checkbox(value=False, description='Display VCR stack', layout={'width': '95%'})
    play_pause_btn = ToggleButton(value=False, icon='play', tooltip='Play/Pause')


    ##################### Filters widget tools #####################
    kernel_size_slider = IntSlider(min=1, max=15, value=3, step=2, description='Kernel Size', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    double_gaussian_slider1 = FloatSlider(min=0.01, max=2.0, value=0.5, step=0.01, description='σ₁', style={'description_width': '60px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    double_gaussian_slider2 = FloatSlider(min=0.01, max=2.0, value=0.2, step=0.01, description='σ₂', style={'description_width': '60px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    double_gaussian_weight_slider = FloatSlider(min=0.1, max=1.0, value=0.5, step=0.01, description='Weight', style={'description_width': '90px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    gaussian_sigma_slider = FloatSlider(min=0.02, max=8.0, value=1.0, step=0.02, description='Gaussian σ', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    gamma_slider = FloatSlider(min=0.02, max=5.0, value=1, step=0.01, description='Gamma', style={'description_width': '90px'}, layout={'display': 'none', 'width': '95%'},continuous_update=False)
    gamma_fft_slider = FloatSlider(min=0.1, max=5.0, value=0.5, step=0.01, description='FFT Gamma', style={'description_width': '100px'}, layout={'width': '95%'})  
    contrast_slider = FloatSlider(min=0.005, max=5.0, value=1.0, step=0.005, description='Contrast', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    gaussian_sigma_fft_slider = FloatSlider(min=0.01, max=1.2, value=0.4, step=0.02, description='FFT σ', style={'description_width': '80px'}, layout={'width': '95%'})

    t = 8  # Default value for CLAHE tile size
    clahe_clip_slider = FloatSlider(min=0.1, max=4.0, value=1, step=0.01, description='CLAHE Clip', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    clahe_tile_slider = IntSlider(min=1, max=42, value=t, step=1, description='CLAHE Tile', layout={'display': 'none', 'width': '95%'}, continuous_update=False)

    brightness_slider = FloatSlider(min=-150, max=250, value=0, step=1, description='Brightness', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    sigmoid_alpha_slider = FloatSlider(min=0, max=20, value=10, step=0.5, description='Sigmoid Alpha', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    sigmoid_beta_slider = FloatSlider(min=0, max=1, value=0.5, step=0.01, description='Sigmoid Beta', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    ################### figures related tools #####################
    scalebar_length_slider = IntSlider(min=1, max=400, value=10, step=1, description='Scalebar length (nm)', style={'description_width': '300px'}, layout={'display': 'none','width': '95%'}, continuous_update=False)
    scalebar_length_text = IntText(value=10, description='Scalebar length (nm):', style={'description_width': '300px'}, layout={'display': 'none', 'width': '95%'})
    scalebar_length_checkbox = Checkbox(value=False, description='Add specific scalebar length (nm)')

    #Selected region sliders
    region_x_slider = IntSlider(min=0, max=2048, value=0, description='X:', layout={'display': 'none', 'width': '195%'})
    region_y_slider = IntSlider(min=0, max=2048, value=0, description='Y:', layout={'display': 'none', 'width': '195%'})
    region_width_text = IntText(value=100, description='Width:', layout={'display': 'none', 'width': '95%'})
    region_height_text = IntText(value=100, description='Height:', layout={'display': 'none', 'width': '95%'})

    dpi_slider = IntSlider(min=50, max=1400, value=100, step=25, description='DPI', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    resize_factor_slider = FloatSlider(min=0.125, max=10, value=1, step=0.125, description='Factor:', style={'description_width': '120px'}, layout={'width': '95%'})
    fig_x_text = FloatText(value=16.00, description='Figure X (cm):', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    fig_y_text = FloatText(value=8.00, description='Figure Y (cm):', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    dtermine_fig_size_checkbox = Checkbox(value=False, description='Determine figure size')

    # Sliders for two-steps thresholding image analysis
    threshold_slider1 = FloatSlider(min=0, max=255, value=100, step=0.1, description='Threshold1', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'},continuous_update=False)
    threshold_slider2 = FloatSlider(min=0, max=255, value=100, step=0.1, description='Threshold2', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    threshold_sa_slider1 = FloatSlider(min=0, max=255, value=50, step=0.1, description='Threshold_sa1', style={'description_width': '120px'}, layout={'display': 'none','width': '95%'}, continuous_update=False)
    threshold_sa_slider2 = FloatSlider(min=0, max=255, value=50, step=0.1, description='Threshold_sa2', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)

    min_clean_cont_area_slider = FloatSlider(min=0, max=10000, value=50, step=0.02, description='Min Clean_Cont Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    max_clean_cont_area_slider = FloatSlider(min=0, max=4200000, value=10000, step=5, description='Max Clean_Cont Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    min_cluster_area_slider = FloatSlider(min=0, max=200, value=0.1, step=0.001, description='Min Clust_sa Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f', continuous_update=False)
    max_cluster_area_slider = FloatSlider(min=0, max=2000, value=1, step=0.005, description='Max Cluster_sa Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f', continuous_update=False)
    min_circularity_slider = FloatSlider(min=0, max=1, value=0.0, step=0.01, description='Min Circularity', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f', continuous_update=False)
    min_isolation_slider = FloatSlider(min=0.002, max=100, value=0.1, step=0.004, description='Min Isolation', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f', continuous_update=False)
    single_atom_clusters_definer_slider = FloatSlider(min=0, max=2, value=0.03, step=0.005, description='SA Cluster definer', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f', continuous_update=False)
    make_circular_thresh_slider = FloatSlider(min=0, max=2, value=0.03, step=0.005, description='Make circular thresh', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})                
    make_circular_thresh_checkbox = Checkbox(value=False, description='Make circular thresh')

    number_of_layers_text = IntText(value=1, description='Number of layers:', style={'description_width': '140px'}, layout={'width': '95%'})
    number_of_layers_button = Button(description="Set Number of Layers", tooltip="Set the number of layers for the analysis", layout={'width': '220px'})
    #############################  Morphological operations widgets #######################################
    opening_slider = IntSlider(min=0, max=10, value=0, description='Opening Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    closing_slider = IntSlider(min=0, max=10, value=0, description='Closing Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    dilation_slider = IntSlider(min=0, max=10, value=0, description='Dilation Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    erosion_slider = IntSlider(min=0, max=10, value=0, description='Erosion Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    gradient_slider = IntSlider(min=0, max=10, value=0, description='Gradient Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    boundary_slider = IntSlider(min=0, max=10, value=0, description='Boundary Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    black_hat_slider = IntSlider(min=0, max=10, value=0, description='Black Hat Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    top_hat_slider = IntSlider(min=0, max=10, value=0, description='Top Hat Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    opening2_slider = IntSlider(min=0, max=10, value=0, description='Opening Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    closing2_slider = IntSlider(min=0, max=10, value=0, description='Closing Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    dilation2_slider = IntSlider(min=0, max=10, value=0, description='Dilation Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    erosion2_slider = IntSlider(min=0, max=10, value=0, description='Erosion Iterations 2', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    gradient2_slider = IntSlider(min=0, max=10, value=0, description='Gradient Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    boundary2_slider = IntSlider(min=0, max=10, value=0, description='Boundary Iterations 2', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)


    ################################## FFT Calibration Sliders #########################################
    min_sampling_slider = FloatSlider(min=0.005, max=2.0, value=0.05, step=0.005, description='Min Samp (nm⁻¹)', style={'description_width': '120px'}, layout={'width': '95%'}, continuous_update=False)
    max_sampling_slider = FloatSlider(min=0.01, max=2.0, value=0.8, step=0.01, description='Max Samp (nm⁻¹)', style={'description_width': '120px'}, layout={'width': '95%'}, continuous_update=False)
    n_widget_slider = IntSlider(min=1, max=1600, value=200, step=1, description='N Points', style={'description_width': '90px'}, layout={'width': '95%'}, continuous_update=False)
    fft_spots_rotation_slider = FloatSlider(min=0, max=360, value=0, step=0.25, description='Rotation (°)',  style={'description_width': '100px'},  layout={'width': '95%'}, continuous_update=False)
    rolloff_slider = FloatSlider(min=0.05, max=2, value=0.33, step=0.05,  description='Rolloff', style={'description_width': '80px'}, layout={'width': '95%'}, continuous_update=False)
    cuttoff_slider = FloatSlider(min=0.05, max=2, value=0.5, step=0.05, description='Cutoff', style={'description_width': '80px'}, layout={'width': '95%'}, continuous_update=False)


    ################################     Checkboxes #########################################
    kernel_size_checkbox = Checkbox(value=False, description='Choose Kernel Size')
    opening_checkbox = Checkbox(value=False, description='Apply Opening')
    closing_checkbox = Checkbox(value=False, description='Apply Closing')
    dilation_checkbox = Checkbox(value=False, description='Apply Dilation')
    erosion_checkbox = Checkbox(value=False, description='Apply Erosion')
    gradient_checkbox = Checkbox(value=False, description='Apply Gradient')
    boundary_checkbox = Checkbox(value=False, description='Apply Boundary Extraction')
    black_hat_checkbox = Checkbox(value=False, description='Apply Black Hat')
    top_hat_checkbox = Checkbox(value=False, description='Apply Top Hat')

    clean_area_analysis_checkbox = Checkbox(value=False, description='Clean area analysis')
    contaminated_area_analysis_checkbox = Checkbox(value=False, description='Contaminated area analysis')

    min_clean_cont_area_checkbox = Checkbox(value=False, description='Min clean_cont area')
    max_clean_cont_area_checkbox = Checkbox(value=False, description='Max clean_cont area')
    min_cluster_area_checkbox = Checkbox(value=False, description='Min cluster_sa area')
    max_cluster_area_checkbox = Checkbox(value=False, description='Max cluster_sa area')
    min_circularity_checkbox = Checkbox(value=False, description='Min Circularity')
    min_isolation_checkbox = Checkbox(value=False, description='Min Isolation')

    opening2_checkbox = Checkbox(value=False, description='Apply Opening2')
    closing2_checkbox = Checkbox(value=False, description='Apply Closing2')
    dilation2_checkbox = Checkbox(value=False, description='Apply Dilation2')
    erosion2_checkbox = Checkbox(value=False, description='Apply Erosion2')
    gradient2_checkbox = Checkbox(value=False, description='Apply Gradient2')
    boundary2_checkbox = Checkbox(value=False, description='Apply Boundary Extraction2')
    single_atom_clusters_definer_checkbox = Checkbox(value=False, description='Single Atom Clusters Definer')
    make_circular_thresh_checkbox = Checkbox(value=False, description='Make circular thresh')
    threshold_checkbox = Checkbox(value=False, description='1st Threshold', layout={'width': '95%'})
    threshold_sa_checkbox = Checkbox(value=False, description='2nd Threshold SA', layout={'width': '95%'})

    # Choose thresholding method
    thresh_method_dropdown = Dropdown(options=['Manual', 'K-means', 'Iterative Otsu'], value='Manual', description='Thresholding Method')
    # K-means Thresholding widget
    kmeans_clusters_number = IntSlider(value=3, min=2, max=10, description='KM Clusters', continuous_update=False)
    kmeans_attempts = IntSlider(value=60, min=1, max=100, description='KM Attempts', continuous_update=False)
    kmeans_epsilon = FloatSlider(value=1.0, min=0.01, max=2.0, step=0.01, description='KM Epsilon', continuous_update=False)
    kmeans_initial_dropdown = Dropdown(options=['K-means++', 'Random'], value='K-means++', description='K-means Initialization')

    # Iterative Otsu Thresholding widget
    val = 3
    iterative_otsu_classes_number= IntSlider(value=val, min=2, max=5, description='Number of Classes', continuous_update=False)
    iterative_otsu_region_selection = IntSlider(value=val - 1, min=1, max=4, description='Select Region', continuous_update=False)

    ########################### Filters Checkboxes ###################################
    contrast_checkbox = Checkbox(value=False, description='Contrast Enhancement')
    gaussian_checkbox = Checkbox(value=False, description='Gaussian Blur')
    double_gaussian_checkbox = Checkbox(value=False, description='Double Gaussian Filter')
    brightness_checkbox = Checkbox(value=False, description='Brightness Adjustment')
    sigmoid_checkbox = Checkbox(value=False, description='Sigmoid Contrast')
    log_transform_checkbox = Checkbox(value=False, description='Log Transform')
    exp_transform_checkbox = Checkbox(value=False, description='Exp Transform')
    clahe_checkbox = Checkbox(value=False, description='Apply CLAHE')
    gamma_checkbox = Checkbox(value=False, description='Gamma Correction')


    ########################## Image Analysis Checkboxes ##########################
    specific_region_checkbox = Checkbox(value=False, description='Select specific region', layout={'width': '95%'})    
    display_specific_region_checkbox = Checkbox(value=False, description='Display specific region', layout={'width': '95%'})

    scalebar_length_checkbox = Checkbox(value=False, description='Custom scalebar', layout={'width': '95%'})
    dpi_checkbox = Checkbox(value=False, description='Custom DPI', layout={'width': '95%'})      
    resize_checkbox = Checkbox(value=False, description='Resize image', layout={'width': '95%'})
    show_plots_checkbox = Checkbox(value=False, description='Show Plots', layout={'width': '95%'})

    ########################## FFT Calibration Checkboxes ##########################
    apply_calibration_checkbox = Checkbox(value=False, description='Apply to All', layout={'width': '95%'})
    gaussian_checkbox = Checkbox(value=False, description='Gaussian', layout={'width': '95%'})
    double_gaussian_checkbox = Checkbox(value=False, description='Double Gaussian', layout={'width': '95%'})
    calibrate_region_checkbox = Checkbox(value=False, description='Custom Region', style={'description_width': '200px'}, layout={'width': '95%'})
    calibration_checkbox = Checkbox(value=False, description='FFT Calibration', layout={'width': '95%'})
    save_for_figure_checkbox = Checkbox(value=False, description='Save Figure', layout={'width': '95%'}) 
    contrast_checkbox = Checkbox(value=False, description='Contrast', layout={'width': '95%'})
    gamma_checkbox = Checkbox(value=False, description='Gamma', layout={'width': '95%'})


    widgets_dict = {}
    for name, value in locals().copy().items():
        widgets_dict[name] = value
    
    return widgets_dict

