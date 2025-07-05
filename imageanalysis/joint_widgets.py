import cv2
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from ipywidgets import (interactive_output, HBox, VBox, FloatSlider, IntSlider, 
                        Checkbox, Button, Output, Dropdown, IntText, FloatText, 
                        Text, HTML, Tab, Accordion, Layout, GridBox)




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



def create_widgets():    
    """
    Create and return a dictionary of widgets used in the image analysis application.
    This function initializes various widgets such as sliders, dropdowns, checkboxes, etc.
    for user interaction in the image analysis process.
    """

    ###################### Dropdowns ######################
    colormap_dropdown = Dropdown(
        options=[ 'gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'bone', 'pink', 'spring', 'summer', 'autumn', 'winter',
        'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'],
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


    ##################### Filters widget tools #####################
    kernel_size_slider = IntSlider(min=1, max=15, value=3, step=2, description='Kernel Size', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    double_gaussian_slider1 = FloatSlider(min=0.05, max=2.0, value=0.5, step=0.01, description='σ₁', style={'description_width': '60px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    double_gaussian_slider2 = FloatSlider(min=0.05, max=2.0, value=0.2, step=0.01, description='σ₂', style={'description_width': '60px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    double_gaussian_weight_slider = FloatSlider(min=0.1, max=1.0, value=0.5, step=0.01, description='Weight', style={'description_width': '90px'}, layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    gaussian_sigma_slider = FloatSlider(min=0.02, max=8.0, value=1.0, step=0.02, description='Gaussian σ', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    gamma_slider = FloatSlider(min=0.02, max=5.0, value=1, step=0.01, description='Gamma', style={'description_width': '90px'}, layout={'display': 'none', 'width': '95%'},continuous_update=False)
    gamma_fft_slider = FloatSlider(min=0.1, max=5.0, value=0.5, step=0.01, description='FFT Gamma', style={'description_width': '100px'}, layout={'width': '95%'})  
    contrast_slider = FloatSlider(min=0.005, max=5.0, value=1.0, step=0.005, description='Contrast', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    gaussian_sigma_fft_slider = FloatSlider(min=0.01, max=1.2, value=0.4, step=0.02, description='FFT σ', style={'description_width': '80px'}, layout={'width': '95%'})

    t = 8  # Default value for CLAHE tile size
    clahe_clip_slider = FloatSlider(min=0.1, max=4.0, value=1, step=0.01, description='CLAHE Clip', layout={'display': 'none', 'width': '95%'}, continuous_update=False)
    clahe_tile_slider = IntSlider(min=1, max=42, value=t, step=1, description='CLAHE Tile', layout={'display': 'none', 'width': '95%'}, continuous_update=False)

    brightness_slider = FloatSlider(min=-150, max=250, value=0, step=1, description='Brightness', layout={'display': 'none', 'width': '95%'})
    sigmoid_alpha_slider = FloatSlider(min=0, max=20, value=10, step=0.5, description='Sigmoid Alpha', layout={'display': 'none', 'width': '95%'})
    sigmoid_beta_slider = FloatSlider(min=0, max=1, value=0.5, step=0.01, description='Sigmoid Beta', layout={'display': 'none', 'width': '95%'})
    ################### figures related tools #####################
    scalebar_length_slider = IntSlider(min=1, max=400, value=10, step=1, description='Scalebar length (nm)', style={'description_width': '300px'}, layout={'display': 'none','width': '95%'})
    scalebar_length_text = IntText(value=10, description='Scalebar length (nm):', style={'description_width': '300px'}, layout={'display': 'none', 'width': '95%'})
    scalebar_length_checkbox = Checkbox(value=False, description='Add specific scalebar length (nm)')

    #Selected region sliders
    region_x_slider = IntSlider(min=0, max=2048, value=0, description='X:', layout={'display': 'none', 'width': '195%'})
    region_y_slider = IntSlider(min=0, max=2048, value=0, description='Y:', layout={'display': 'none', 'width': '195%'})
    region_width_text = IntText(value=100, description='Width:', layout={'display': 'none', 'width': '95%'})
    region_height_text = IntText(value=100, description='Height:', layout={'display': 'none', 'width': '95%'})

    dpi_slider = IntSlider(min=100, max=1200, value=300, step=100, description='DPI', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    resize_factor_slider = IntSlider(min=1, max=10, value=1, step=1, description='Factor:', style={'description_width': '120px'}, layout={'width': '95%'})
    fig_x_text = IntText(value=16, description='Figure X (cm):', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    fig_y_text = IntText(value=8, description='Figure Y (cm):', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'})
    dtermine_fig_size_checkbox = Checkbox(value=False, description='Determine figure size')

    # Sliders for two-steps thresholding image analysis
    threshold_slider = FloatSlider(min=0, max=255, value=100, step=0.2, description='Threshold', style={'description_width': '120px'}, layout={'display': 'none', 'width': '95%'},continuous_update=False)
    threshold_slider_sa = FloatSlider(min=0, max=255, value=100, step=0.5, description='Threshold_sa', style={'description_width': '120px'}, layout={'width': '95%'}, continuous_update=False)

    min_area_slider = FloatSlider(min=0, max=100, value=50, step=0.02, description='Min Clean_Cont Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    max_area_slider = FloatSlider(min=0, max=120000, value=10000, step=5, description='Max Clean_Cont Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    min_area_sa_clusters = FloatSlider(min=0, max=200, value=0.1, step=0.001, description='Min Clust_sa Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f')
    max_area_sa_clusters = FloatSlider(min=0, max=2000, value=1, step=0.005, description='Max Cluster_sa Area', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f')
    circularity_slider = FloatSlider(min=0, max=1, value=0.5, step=0.01, description='Min Circularity', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f')
    min_isolation_slider = FloatSlider(min=0.002, max=100, value=1, step=0.004, description='Min Isolation', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f')
    single_atom_clusters_definer = FloatSlider(min=0, max=2, value=0.5, step=0.005, description='SA Cluster definer', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'}, readout_format='.3f')
    make_circular_thresh = FloatSlider(min=0, max=2, value=0.03, step=0.005, description='Make circular thresh', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})                


    #############################  Morphological operations widgets #######################################
    opening_slider = IntSlider(min=0, max=10, value=0, description='Opening Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    closing_slider = IntSlider(min=0, max=10, value=0, description='Closing Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    dilation_slider = IntSlider(min=0, max=10, value=0, description='Dilation Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    erosion_slider = IntSlider(min=0, max=10, value=0, description='Erosion Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    gradient_slider = IntSlider(min=0, max=10, value=0, description='Gradient Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    boundary_slider = IntSlider(min=0, max=10, value=0, description='Boundary Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    black_hat_slider = IntSlider(min=0, max=10, value=0, description='Black Hat Iterations',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    top_hat_slider = IntSlider(min=0, max=10, value=0, description='Top Hat Iterations', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    opening_slider2 = IntSlider(min=0, max=10, value=0, description='Opening Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    closing_slider2 = IntSlider(min=0, max=10, value=0, description='Closing Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    dilation_slider2 = IntSlider(min=0, max=10, value=0, description='Dilation Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    erosion_slider2 = IntSlider(min=0, max=10, value=0, description='Erosion Iterations 2', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    gradient_slider2 = IntSlider(min=0, max=10, value=0, description='Gradient Iterations 2',style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})
    boundary_slider2 = IntSlider(min=0, max=10, value=0, description='Boundary Iterations 2', style={'description_width': '160px'}, layout={'display': 'none', 'width': '95%'})


    ################################## FFT Calibration Sliders #########################################
    min_sampling_slider = FloatSlider(min=0.005, max=2.0, value=0.05, step=0.005, description='Min Samp (nm⁻¹)', style={'description_width': '120px'}, layout={'width': '95%'})
    max_sampling_slider = FloatSlider(min=0.01, max=2.0, value=0.8, step=0.01, description='Max Samp (nm⁻¹)', style={'description_width': '120px'}, layout={'width': '95%'})
    n_widget_slider = IntSlider(min=1, max=1600, value=200, step=1, description='N Points', style={'description_width': '90px'}, layout={'width': '95%'})
    fft_spots_rotation_slider = FloatSlider(min=0, max=360, value=0, step=1, description='Rotation (°)',  style={'description_width': '100px'},  layout={'width': '95%'})
    rolloff_slider = FloatSlider(min=0.05, max=2, value=0.33, step=0.05,  description='Rolloff', style={'description_width': '80px'}, layout={'width': '95%'})
    cuttoff_slider = FloatSlider(min=0.05, max=2, value=0.5, step=0.05, description='Cutoff', style={'description_width': '80px'}, layout={'width': '95%'})


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

    min_area_checkbox = Checkbox(value=False, description='Min clean_cont area')
    max_area_checkbox = Checkbox(value=False, description='Max clean_cont area')
    min_area_checkbox_sa = Checkbox(value=False, description='Min cluster_sa area')
    max_area_checkbox_sa = Checkbox(value=False, description='Max cluster_sa area')
    circularity_checkbox = Checkbox(value=False, description='Min Circularity')
    isolation_checkbox = Checkbox(value=False, description='Min Isolation')

    opening_checkbox2 = Checkbox(value=False, description='Apply Opening2')
    closing_checkbox2 = Checkbox(value=False, description='Apply Closing2')
    dilation_checkbox2 = Checkbox(value=False, description='Apply Dilation2')
    erosion_checkbox2 = Checkbox(value=False, description='Apply Erosion2')
    gradient_checkbox2 = Checkbox(value=False, description='Apply Gradient2')
    boundary_checkbox2 = Checkbox(value=False, description='Apply Boundary Extraction2')
    single_atom_clusters_definer_checkbox = Checkbox(value=False, description='Single Atom Clusters Definer')
    make_circular_thresh_checkbox = Checkbox(value=False, description='Make circular thresh')


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
    specific_region_checkbox = Checkbox(value=False, description='Save specific region', layout={'width': '95%'})
    scalebar_length_checkbox = Checkbox(value=False, description='Custom scalebar', layout={'width': '95%'})
    dpi_checkbox = Checkbox(value=False, description='Custom DPI', layout={'width': '95%'})      
    resize_checkbox = Checkbox(value=False, description='Resize image', layout={'width': '95%'})
    show_plots_checkbox = Checkbox(value=False, description='Show Plots', layout={'width': '95%'})

    ########################## FFT Calibration Checkboxes ##########################
    apply_calibration_checkbox = Checkbox(value=False, description='Apply to All', layout={'width': '95%'})
    gaussian_checkbox = Checkbox(value=False, description='Gaussian', layout={'width': '95%'})
    double_gaussian_checkbox = Checkbox(value=False, description='Double Gaussian', layout={'width': '95%'})
    calibrate_region_checkbox = Checkbox(value=False, description='Custom Region', layout={'width': '95%'})
    calibration_checkbox = Checkbox(value=False, description='FFT Calibration', layout={'width': '95%'})
    save_for_figure_checkbox = Checkbox(value=False, description='Save Figure', layout={'width': '95%'}) 
    contrast_checkbox = Checkbox(value=False, description='Contrast', layout={'width': '95%'})
    gamma_checkbox = Checkbox(value=False, description='Gamma', layout={'width': '95%'})


    widgets_dict = {}
    for name, value in locals().copy().items():
        if name.endswith(('_dropdown', '_slider', '_checkbox', '_slider1', '_slider2', '_slider_sa', 'text')):
            widgets_dict[name] = value
    
    return widgets_dict

