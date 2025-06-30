import os
import json
import h5py
import re
from imagesequence import ImageSequence

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
            
if __name__ == "__main__":
    # Example usage
    save_eels_pairs_hdf5('/home/somar/Desktop/2025/Data for publication/Sample 2344/EELS/')
