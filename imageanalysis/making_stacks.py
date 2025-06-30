import os
import re
import json
import h5py
import joblib
from data_loader import ImageSequence


##################### prepare the metadata to be json serializable #####################
def dict_to_json_serializable(d):
    """Recursively convert non-serializable values to strings"""
    def _convert(value):
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        else:
            return str(value)  # Convert non-serializable objects to strings
    return _convert(d)



###################### save stack as HDF5 #####################
def save_stack_hdf5(directory, output_file="stacktest.h5"):
    with h5py.File(os.path.join(directory, output_file), "w") as hf:
        img_group = hf.create_group("images")
        meta_group = hf.create_group("metadata")

        valid_files = sorted([
            f for f in os.listdir(directory) 
            if f.endswith('.ndata1') and 'SuperScan (HAADF) (Gaussian Blur)' not in f
        ])

        fov_data = []

        for idx, filename in enumerate(valid_files):
            file_path = os.path.join(directory, filename)
            imageseq = ImageSequence(file_path)

            try:
                raw_meta = imageseq.raw_metadata or {}
            except AttributeError:
                raw_meta = {}

            try:
                metadata_dict = raw_meta[0] if hasattr(raw_meta, '__getitem__') else raw_meta
            except Exception:
                metadata_dict = {}

            metadata = metadata_dict.get('metadata', {})

            # Extract FOV with fallbacks
            fov = float('inf')
            try:
                fov = float(metadata.get('scan', {}).get('scan_device_properties', {}).get('fov_nm', float('inf')))
            except Exception:
                try:
                    fov = float(metadata_dict.get('instrument', {}).get('ImageScanned', {}).get('fov_nm', float('inf')))
                except Exception:
                    pass


            try:
                img = imageseq.raw_data[0] 
            except Exception:
                continue

            fov_data.append((idx, fov, img, metadata_dict))

        # Sort by FOV (lowest first; unknown at end)
        fov_data.sort(key=lambda x: x[1])

        for new_idx, (orig_idx, fov, img, meta) in enumerate(fov_data):
            # Store image
            img_ds = img_group.create_dataset(
                name=f"image_{new_idx:04d}",
                data=img,
                compression="gzip"
            )
            img_ds.attrs["fov_nm"] = fov if fov != float('inf') else "unknown"

            # Store metadata
            meta_ds = meta_group.create_dataset(
                name=f"metadata_{new_idx:04d}",
                data=json.dumps(dict_to_json_serializable(meta)),
                dtype=h5py.string_dtype()
            )

            # Cross-reference
            img_ds.attrs["metadata_ref"] = f"metadata_{new_idx:04d}"
            meta_ds.attrs["image_ref"] = f"image_{new_idx:04d}"
            meta_ds.attrs["original_index"] = orig_idx

        hf.attrs["sorting"] = "fov_nm (unknown FOV at end)"
        hf.attrs["sorting_version"] = "1.0"


def process_directory_h5(directory, output_file="stack.h5"):
    # Always process subdirectories first
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            process_directory_h5(subdir_path, output_file)
    
    # Then process current directory if it has .ndata1 files
    if any(f.endswith('.ndata1') for f in os.listdir(directory)):
        save_stack_hdf5(directory, output_file)


####################### save stack as pickle #####################
def save_stack_images_pkl(directory, metadata=False):
    images = []
    metadata_list = []
    output_file_path = directory + '/' + 'stack.pkl'
    metadata_file_path = directory + '/' + 'metadata.pkl'
    # Load images while preserving original shapes
    for file in os.listdir(directory):
        if file.endswith('.ndata1') and 'SuperScan (HAADF) (Gaussian Blur)' not in file:
            imageseq = ImageSequence(os.path.join(directory, file))
            image = imageseq.raw_data  # Assuming this is a NumPy array
            metadata = imageseq.raw_metadata  # Assuming metadata is a dictionary
            images.append(image)
            metadata_list.append(metadata)
    
    if images:  # Only save if there are images
        joblib.dump(images, output_file_path)
        joblib.dump(metadata_list, metadata_file_path)



def process_directory_pkl(directory):
    has_files = any(file.endswith('.ndata1') for file in os.listdir(directory))
    
    if has_files:
        save_stack_images_pkl(directory)
    else:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                process_directory_pkl(subdir_path)




######################## save EELS pairs as HDF5 #####################
def extract_point_id(filename):
    """Extract point ID from filename using regex (e.g., #8877)"""
    match = re.search(r'#(\d+)', filename)
    return match.group(1) if match else None

def find_matching_image(directory, point_id):
    """Find MAADF/HAADF image matching the point ID"""
    for f in os.listdir(directory):
        if (f.startswith('MAADF') or f.startswith('HAADF')) and f'#{point_id}' in f:
            return f
    return None

def save_eels_pairs_hdf5(directory, output_file="eels_pairs.h5"):
    with h5py.File(os.path.join(directory, output_file), "w") as hf:
        pairs_group = hf.create_group("pairs")
        
        # Find all point spectrum files
        point_files = [f for f in os.listdir(directory) 
                      if f.startswith('Point spectrum') and f.endswith('.ndata1')]
        
        for idx, pt_file in enumerate(point_files):
            point_id = extract_point_id(pt_file)
            if not point_id:
                print(f"Skipping {pt_file}: No point ID found")
                continue
                
            # Find matching image
            img_file = find_matching_image(directory, point_id)
            if not img_file:
                print(f"Skipping {pt_file}: No matching image found")
                continue
                
            pt_path = os.path.join(directory, pt_file)
            img_path = os.path.join(directory, img_file)
            
            # Process point spectrum
            try:
                pt_seq = ImageSequence(pt_path)
                pt_data = pt_seq.raw_data.squeeze()  # Convert to 1D
                pt_meta = pt_seq.raw_metadata or {}
            except Exception as e:
                print(f"Error reading {pt_file}: {str(e)}")
                continue
                
            # Process image
            try:
                img_seq = ImageSequence(img_path)
                img_data = img_seq.raw_data.squeeze()  
                img_meta = img_seq.raw_metadata or {}
            except Exception as e:
                print(f"Error reading {img_file}: {str(e)}")
                continue
                
            # Create pair group
            pair_group = pairs_group.create_group(f"pair_{idx:04d}")
            
            # Store original filenames
            pair_group.attrs['point_spectrum_file'] = pt_file
            pair_group.attrs['image_file'] = img_file
            
            # Store data
            pair_group.create_dataset("spectrum", data=pt_data, compression="gzip")
            pair_group.create_dataset("image", data=img_data, compression="gzip")
            
            # Store metadata
            pt_meta_processed = json.dumps(dict_to_json_serializable(pt_meta))
            img_meta_processed = json.dumps(dict_to_json_serializable(img_meta))
            
            pair_group.create_dataset(
                "spectrum_metadata", 
                data=pt_meta_processed,
                dtype=h5py.string_dtype()
            )
            pair_group.create_dataset(
                "image_metadata", 
                data=img_meta_processed,
                dtype=h5py.string_dtype())


######################### Example usage of the script ########################
if __name__ == "__main__":
    # process_directory_pkl('/home/somar/Desktop/2025/Data for publication/Sample 2344/ADF images/')
    process_directory_h5('/home/somar/Desktop/2025/Data for publication/Sample 2344/ADF images/After_Heating_200C/', output_file="test_stack.h5")
    # process_directory_h5('/home/somar/Desktop/2025/Data for publication/Multilayer graphene/')    
    # save_eels_pairs_hdf5('/home/somar/Desktop/2025/Data for publication/Sample 2344/EELS/')
