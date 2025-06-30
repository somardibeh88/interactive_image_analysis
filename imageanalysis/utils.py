import os
import cv2
import joblib
import numpy as np
from filters import double_gaussian
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from skimage import measure, exposure, filters
from data_loader import ImageSequence
from ipywidgets import interactive_output, IntSlider, FloatSlider, HBox, VBox
import ipywidgets as widgets


def process_image_with_scale_bar(image, fov, dpi, output_path, font_path):
    """
    Process an image, overlay a scale bar, and save the output as a PNG file.

    Parameters:
    - image: np.ndarray, the input image (assumed to be normalized if CLAHE is applied).
    - fov: float, field of view in nanometers.
    - dpi: int, resolution for saving the PNG file.
    - output_path: str, file path to save the image with the scale bar.
    - font_path: str, path to the font file for the scale bar text.

    Returns:
    - None
    """
    # Ensure image is normalized (0 to 1)
    if image.max() > 1:
        image = image / image.max()

    # Convert the image to 8-bit
    image_8bit = (image * 255).astype(np.uint8)

    # Apply a colormap to the image
    colored_image = cv2.applyColorMap(image_8bit, cv2.COLORMAP_MAGMA)

    # Image parameters
    image_size = image.shape[0]  # Assuming square image
    shift = int(image_size / 16)

    # Scale calculations
    scale_per_pixel = fov / image_size  # nm/pixel
    desired_scale_bar_length_nm = int(fov / 4)  # Desired scale bar length in nm
    scale_bar_length_pixels = int(desired_scale_bar_length_nm / scale_per_pixel)

    # Scale bar position and dimensions
    start_point = (shift, image_size - int(shift * 0.66))
    end_point = (start_point[0] + scale_bar_length_pixels, start_point[1])
    scale_bar_thickness = max(1, int(image_size / (6 * fov)))

    # Convert the image to a PIL Image for scale bar and text drawing
    colored_image_pil = Image.fromarray(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(colored_image_pil)

    # Draw the scale bar
    scale_bar_color = (255, 255, 255)  # White color
    draw.line([start_point, end_point], fill=scale_bar_color, width=scale_bar_thickness)

    # Draw the scale bar text
    font = ImageFont.truetype(font_path, 112)  # Adjust font size if necessary
    text = f'{desired_scale_bar_length_nm} nm'
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_origin = (
        start_point[0] + (scale_bar_length_pixels - text_width) / 2,
        start_point[1] - int(shift * 0.5) - text_height,
    )
    draw.text(text_origin, text, fill=scale_bar_color, font=font)

    # Save the image with the scale bar
    colored_image_pil.save(output_path, format='PNG', dpi=(dpi, dpi))




def eels_analysis(spectrum_image, background_image, ticks_density, smoothing_params, save = False):
    """
    Perform EELS analysis with background subtraction and smoothing.
    
    Parameters:
        spectrum_image (ImageSequence): The EELS spectrum image.
        background_image (ImageSequence): The background spectrum image.
        ticks_density (int): Density of ticks for plotting axes.
        smoothing_params (dict): Parameters for smoothing, e.g., {'window_length': 15, 'polyorder': 3}.
    
    Returns:
        None
    """
    # Extract data and metadata
    spectrum_data = spectrum_image.raw_data - background_image.raw_data
    metadata = spectrum_image.raw_metadata

    # Extract calibration parameters
    energy_offset = metadata['spatial_calibrations'][0]['offset']
    energy_scale = metadata['spatial_calibrations'][0]['scale']

    # Generate energy axis
    energy_axis = energy_offset + energy_scale * np.arange(len(spectrum_data))
    def normalize_to_range(data, ymin, ymax, xmin, xmax):
        """
        Normalize a dataset from [ymin, ymax] to [xmin, xmax].
        
        Parameters:
            data (array-like): The dataset to normalize.
            ymin (float): The minimum value in the original range.
            ymax (float): The maximum value in the original range.
            xmin (float): The minimum value in the target range.
            xmax (float): The maximum value in the target range.
        
        Returns:
            numpy.ndarray: The normalized dataset in the range [xmin, xmax].
        """
        return xmin + (data - ymin) * (xmax - xmin) / (ymax - ymin)
            
    spectrum_data_normalized = normalize_to_range(spectrum_data, min(spectrum_data), max(spectrum_data), min(energy_axis), max(energy_axis))


    # Apply Savitzky-Golay filter for smoothing
    window_length = smoothing_params.get('window_length', 15)
    polyorder = smoothing_params.get('polyorder', 3)

    # Ensure window_length is odd and does not exceed the data length
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(spectrum_data))

    smoothed_spectrum = savgol_filter(spectrum_data_normalized, window_length=window_length, polyorder=polyorder)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot smoothed data
    ax.plot(energy_axis, smoothed_spectrum, linewidth=2)

    ax.set_xlabel('Energy Loss [eV]', fontsize=18)
    ax.set_ylabel('Intensity [a.u.]', fontsize=18)
    ax.set_xlim(left=int(min(energy_axis)), right=max(energy_axis) + 0.01 * max(energy_axis))
    ax.set_ylim(bottom=int(min(spectrum_data_normalized)), top=max(spectrum_data_normalized) + 0.01 * max(spectrum_data_normalized))


    # Set ticks for axes
    major_ticks_x = np.arange(
        int(min(energy_axis)),
        max(energy_axis) + 0.05 * max(energy_axis),
        int(max(energy_axis) /ticks_density),
    )
    major_ticks_y = np.arange(
        int(min(spectrum_data_normalized)),
        max(spectrum_data_normalized) + 0.05 * max(spectrum_data_normalized),
        int(max(spectrum_data_normalized)/ ticks_density),
    )

    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.grid(which='both')
    ax.grid(which='minor', linestyle='--', color='k', alpha=0.2)
    ax.grid(which='major', linestyle='--', color='k', alpha=0.4)
    ax.set_yticklabels([])  # Hide the y-axis labels
    # Add legend
    # ax.legend(fontsize=12, loc='upper right')

    # Save and show the plot
    if save:
        plt.savefig('smoothed_spectrum.png', dpi=1200)
    plt.tight_layout()
    plt.show()




def eels_analysis_all_one_plot(spectrum_image, background_image, ticks_density, smoothing_params, save=False):
    """
    Perform EELS analysis with background subtraction and smoothing.
    
    Parameters:
        spectrum_image (ImageSequence): The EELS spectrum image.
        background_image (ImageSequence): The background spectrum image.
        ticks_density (int): Density of ticks for plotting axes.
        smoothing_params (dict): Parameters for smoothing, e.g., {'window_length': 15, 'polyorder': 3}.
        save (bool): Whether to save the resulting plot as an image.
    
    Returns:
        None
    """

    # Extract data and metadata
    signal_data = spectrum_image.raw_data
    background_data = background_image.raw_data
    spectrum_data = signal_data - background_data
    metadata = spectrum_image.raw_metadata

    # Extract calibration parameters
    energy_offset = metadata['spatial_calibrations'][0]['offset']
    energy_scale = metadata['spatial_calibrations'][0]['scale']

    # Generate energy axis
    energy_axis = energy_offset + energy_scale * np.arange(len(spectrum_data))

    # Apply Savitzky-Golay filter for smoothing
    window_length = smoothing_params.get('window_length', 15)
    polyorder = smoothing_params.get('polyorder', 3)

    # Ensure window_length is odd and does not exceed the data length
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(spectrum_data))

    smoothed_spectrum = savgol_filter(spectrum_data, window_length=window_length, polyorder=polyorder)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the signal, background, and background-subtracted data
    ax.plot(energy_axis, signal_data, label='Signal', color='green', linewidth=2)
    ax.plot(energy_axis, background_data, label='Background', color='red', linestyle='--', linewidth=2)
    # ax.plot(energy_axis, spectrum_data_normalized, label='Background-Subtracted Signal', color='green', linewidth=2)

    # Add smoothed signal
    ax.plot(energy_axis, smoothed_spectrum, label='Smoothed Signal', color='blue', linewidth=2, linestyle='-')

    # Set axis labels and title
    ax.set_xlabel('Energy Loss [eV]', fontsize=18)
    ax.set_ylabel('Intensity [a.u.]', fontsize=18)
    ax.set_title('EELS Spectrum Analysis', fontsize=20)

    # Set axis limits
    ax.set_xlim(left=int(min(energy_axis)), right=max(energy_axis) + int(0.01 * max(energy_axis)))
    ax.set_ylim(bottom= -200 , top=int(max(signal_data)) + int(0.01 * max(signal_data)))  

    # Set ticks for axes
    major_ticks_x = np.arange(
        int(min(energy_axis)),
        max(energy_axis) + 0.05 * max(energy_axis),
        int(max(energy_axis) / ticks_density),
    )

    major_ticks_y = np.arange(
        int(min(background_data)),
        max(signal_data) + 0.05 * max(signal_data),
        int(max(signal_data)/ ticks_density),
    )


    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.grid(which='both')
    ax.grid(which='minor', linestyle='--', color='k', alpha=0.2)
    ax.grid(which='major', linestyle='--', color='k', alpha=0.4)

    # Add legend
    ax.legend(fontsize=14, loc='upper right')

    # Save and show the plot
    if save:
        plt.savefig('eels_analysis.png', dpi=1200)
    plt.tight_layout()
    plt.show()


# Normalize function
def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    return normalized_image.astype(np.uint8)


def improve_contrast(image,percentile=10):
    low = np.percentile(image, percentile)
    high = np.percentile(image,100 - percentile)
    return exposure.rescale_intensity(image, in_range=(low, high))


def save_stack_images_pkl(directory, metadata=False):
    images = []
    metadata_list = []
    if '_' in directory:
        output_file_path = directory + '/' + '_'.join(directory.split('/')[-1].split('_')[1:]) + '.pkl'
    else:
        output_file_path = directory + '/' + 'stack.pkl'
    metadata_file_path = directory + '/' + 'metadata.pkl'
    # Load images while preserving original shapes
    for file in os.listdir(directory):
        if file.endswith('.ndata1'):
            imageseq = ImageSequence(os.path.join(directory, file))
            image = imageseq.raw_data  # Assuming this is a NumPy array
            metadata = imageseq.raw_metadata  # Assuming metadata is a dictionary
            images.append(image)
            metadata_list.append(metadata)
    
    if images:  # Only save if there are images
        joblib.dump(images, output_file_path)
        joblib.dump(metadata_list, metadata_file_path)


def process_directory(directory):
    has_files = any(file.endswith('.ndata1') for file in os.listdir(directory))
    
    if has_files:
        save_stack_images_pkl(directory)
    else:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                process_directory(subdir_path)


# Read the stacked images from the pkl file
def read_stacked_images(file_path):
    images = joblib.load(file_path)
    # print(f"Loaded {len(images)} images from {file_path}")
    return images


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Ensure image is in 8-bit format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def apply_gamma_correction(image, gamma=1.5):
    """
    Apply Gamma Correction to selectively brighten darker regions.
    
    Parameters:
        image (numpy array): Grayscale input image.
        gamma (float): Gamma value (>1 brightens dark regions).
    
    Returns:
        gamma_corrected (numpy array): Brightness-adjusted image.
    """
    # Normalize to range 0-1, apply gamma correction, and rescale to 0-255
        # Ensure image is in 8-bit format
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)




def process_image(image, normalized=False, percentile:float=None, double_gaussian_parameters:dict=None, clahe_parameters:dict=None, gamma_parameters:dict=None):
    if normalized:
        image = normalize(image)
    if percentile is not None:
        image = improve_contrast(image, percentile)
    if double_gaussian_parameters is not None:
        image = double_gaussian(image, **double_gaussian_parameters)
    if clahe_parameters is not None:
        image = apply_clahe(image, **clahe_parameters)
    if gamma_parameters is not None:
        image = apply_gamma_correction(image, **gamma_parameters)
    return image





def interactive_image_analysis(stack, percentile=0.1, kernel=None, nm_per_pixel=1, 
                         minimum_area_nm2=0, max_area_nm2=0.16, fram_area_nm2=1):
    """
    Interactive widget combining stack slice selection and contour analysis with threshold adjustment.
    
    Parameters:
    - stack: List of images (3D numpy array).
    - percentile: Percentile for normalization (default 0.1).
    - kernel: Morphology kernel (default 3x3 ones).
    - nm_per_pixel: Scale conversion (pixels to nm).
    - minimum_area_nm2: Minimum contour area threshold in nm².
    - max_area_nm2: Maximum contour area threshold in nm².
    - fram_area_nm2: Frame area for defect density calculation.
    """
    if kernel is None:
        kernel = np.ones((3, 3), np.uint8)
    
    # Widgets
    slice_slider = widgets.IntSlider(
        min=0, max=len(stack)-1, value=0, description='Slice'
    )
    threshold_slider = widgets.FloatSlider(
        min=0, max=255, value=100, step=1, description='Threshold'
    )
    
    def update(slice_number, threshold):
        image = stack[slice_number]
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (1, 1), 1)
        
        # Process a single image
        image = process_image(blurred, normalized=True, percentile = percentile, double_gaussian_parameters=double_gaussian_parameters,
                              clahe_parameters=clahe_parameters, gamma_parameters=gamma_parameters)
        
        # Thresholding
        _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        eroded = cv2.erode(thresh, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        morphed = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Contour detection
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        areas = []
        ferets = []
        for cnt in contours:
            area_pixel = cv2.contourArea(cnt)
            area_nm2 = area_pixel * (nm_per_pixel ** 2)
            if minimum_area_nm2 < area_nm2 < max_area_nm2:
                valid_contours.append(cnt)
                areas.append(area_nm2)
                if len(cnt) > 5:  # Sufficient points for ellipse fitting
                    (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                    feret = max(MA, ma) * nm_per_pixel
                    ferets.append(feret)
        
        # Create contour mask
        contour_mask = np.zeros_like(morphed)
        cv2.drawContours(contour_mask, valid_contours, -1, 255, thickness=2)
        
        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        axs[0].imshow(image, cmap='gray')
        axs[0].set_title('Normalized Image')
        axs[0].axis('off')
        
        axs[1].imshow(contour_mask, cmap='gray')
        axs[1].set_title('Detected Contours')
        axs[1].axis('off')
        
        axs[2].hist(areas, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axs[2].set_xlabel('Area (nm²)')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Size Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        if areas:
            print(f"Total Area: {sum(areas):.2f} nm²")
            if ferets:
                print(f"Avg Feret Diameter: {np.mean(ferets):.2f} nm")
            print(f"Avg Defect Area: {np.mean(areas):.2f} nm²")
            print(f"Defect Density: {len(areas)/fram_area_nm2*100:.2f} defects/100nm²\n")
        else:
            print("No valid contours found.\n")
    
    # Link widgets to function
    out = interactive_output(update, {'slice_number': slice_slider, 'threshold': threshold_slider})
    
    # Display layout
    controls = HBox([slice_slider, threshold_slider])
    display(VBox([controls, out]))



# Example usage
if __name__ == "__main__":
    # Create a dummy normalized image for testing
    test_image = np.random.rand(512, 512)

    # Parameters
    fov_nm = 2000  # Field of view in nanometers
    dpi_val = 300  # Resolution
    output_file_path = "image_with_scale_bar.png"
    font_file_path = "/path/to/your/font.ttf"  # Replace with a valid path

    process_image_with_scale_bar(test_image, fov_nm, dpi_val, output_file_path, font_file_path)

