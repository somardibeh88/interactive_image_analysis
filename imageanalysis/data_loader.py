""""""""""""""""""""""'""""""
Data Loader Module

Author: Somar Dibeh
"""""""""""""""""""""""""""""

import json
import h5py
import joblib
import zipfile
import tifffile
import numpy as np


# ================== Metadata class for reading the metadata and inspecting it based on a certain given keys ==================
class MetadataMethods:
    def get_metadata(self, target_key, data=None):
        """This method gets all the values of a certain target key in the metadata json file."""
        extracted_values = []
        if data is None:
            data = self._get_full_metadata()
        if isinstance(data, dict):
            if target_key in data and data[target_key] is not None:
                value_to_extract = data.get(target_key)
                extracted_values.append(value_to_extract)
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    extracted_values.extend(self.get_metadata(target_key, value))
        elif isinstance(data, list):
            for item in data:
                extracted_values.extend(self.get_metadata(target_key, item))
        
        if len(extracted_values) == 1 and isinstance(extracted_values[0], (list)):
            return extracted_values[0]
        return extracted_values

    def get_specific_metadata(self, target_key, required_keys=None, data=None, under_required_keys=False):
        """
        This method gets specific metadata based on a condition that the target key is extracted only 
        when under specific required keys in the nested dictionaries.
        """
        if data is None:
            data = self._get_full_metadata()
        if required_keys is None:
            extracted_values = self.get_metadata(target_key, data)
        else:
            extracted_values = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in required_keys:
                        if isinstance(value, dict):
                            extracted_values.extend(self.get_specific_metadata(target_key, required_keys, value, True))
                        elif isinstance(value, list):
                            for item in value:
                                extracted_values.extend(self.get_specific_metadata(target_key, required_keys, item, True))
                    elif under_required_keys and key == target_key:
                        extracted_values.append(value)
                    else:
                        if isinstance(value, (dict, list)):
                            extracted_values.extend(self.get_specific_metadata(target_key, required_keys, value, under_required_keys))
            elif isinstance(data, list):
                for item in data:
                    extracted_values.extend(self.get_specific_metadata(target_key, required_keys, item, under_required_keys))

        if len(extracted_values) == 1 and isinstance(extracted_values[0], (list)):
            return extracted_values[0]
        return extracted_values
    
    

# ================== Unified Loader for different file formats ==================
class UnifiedLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.format = None
        self.raw_data = None
        self.raw_metadata = None
        self._determine_format()
        self._load_data()
        
    def _determine_format(self):
        """Determine file format based on extension"""
        if self.file_path.endswith('.h5'):
            self.format = 'h5'
        elif self.file_path.endswith(('.tif', '.tiff')):
            self.format = 'tiff'
        elif self.file_path.endswith(('.ndata1', '.ndata')):
            self.format = 'ndata'
        elif self.file_path.endswith('.pkl'):
            self.format = 'pkl'
        else:
            raise ValueError(f'Unsupported file format: {self.file_path}')
    
    def _load_data(self):
        """Load data based on file format"""
        if self.format == 'h5':
            self._load_h5()
        elif self.format == 'tiff':
            self._load_tiff()
        elif self.format == 'ndata':
            self._load_ndata()
        elif self.format == 'pkl':
            self._load_pkl()
        
    def _load_h5(self):
        """Lazy loader for HDF5 files (EELS or standard image stack)"""
        self.h5_file = h5py.File(self.file_path, 'r')
        
        if 'pairs' in self.h5_file:
            # EELS structure detected
            class PairLoader(MetadataMethods):
                def __init__(self, h5_file, key):
                    self.h5_file = h5_file
                    self.key = key
                    self.pairs = sorted(h5_file['pairs'].keys())
                
                def __getitem__(self, index):
                    pair_group = self.h5_file['pairs'][self.pairs[index]]
                    item = pair_group[self.key]
                    if self.key.endswith('metadata'):
                        return json.loads(item.asstr()[()])
                    return item[:]
                
                def __len__(self):
                    return len(self.pairs)
                
                def _get_full_metadata(self):
                    return [self[i] for i in range(len(self))]
            
            self.raw_data = PairLoader(self.h5_file, 'image')
            self.raw_metadata = PairLoader(self.h5_file, 'image_metadata')

        elif 'images' in self.h5_file and 'metadata' in self.h5_file:
            # Standard image stack with metadata
            class H5ImageLoader:
                def __init__(self, h5_file):
                    self.group = h5_file['images']
                def __getitem__(self, index):
                    return self.group[f"image_{index:04d}"][:]
                def __len__(self):
                    return len(self.group)

            class H5MetadataLoader(MetadataMethods):
                def __init__(self, h5_file):
                    self.group = h5_file['metadata']
                    # Extract keys and sort by index
                    self.keys = sorted(self.group.keys(), key=lambda x: int(x.split('_')[-1]))
                
                def __getitem__(self, index):
                    if isinstance(index, int):
                        key = self.keys[index]
                    elif isinstance(index, str):
                        key = index
                    else:
                        raise TypeError("Index must be int or str")
                    return json.loads(self.group[key].asstr()[()])
                
                def __len__(self):
                    return len(self.group)
                
                def _get_full_metadata(self):
                    return [self[i] for i in range(len(self))]
            
            self.raw_data = H5ImageLoader(self.h5_file)
            self.raw_metadata = H5MetadataLoader(self.h5_file)

        else:
            raise ValueError("Unrecognized HDF5 structure: neither EELS nor image/metadata group found.")

    def _load_tiff(self):
        """Lazy loader for TIFF files"""
        self.tiff_file = tifffile.TiffFile(self.file_path)
        
        # lazy accessors for images and metadata
        class TiffImageLoader:
            def __init__(self, tiff_file):
                self.tiff_file = tiff_file
                
            def __getitem__(self, index):
                return self.tiff_file.pages[index].asarray()
            
            def __len__(self):
                return len(self.tiff_file.pages)
        
        class TiffMetadataLoader(MetadataMethods):
            def __init__(self, tiff_file):
                self.tiff_file = tiff_file
                
            def __getitem__(self, index):
                try:
                    desc = self.tiff_file.pages[index].tags['ImageDescription'].value
                    if 'nion.1=' in desc:
                        json_str = desc.split('nion.1=')[1]
                        return json.loads(json_str)
                    return json.loads(desc)
                except (KeyError, json.JSONDecodeError):
                    return {}
            
            def __len__(self):
                return len(self.tiff_file.pages)
            
            def _get_full_metadata(self):
                return [self[i] for i in range(len(self))]
        
        self.raw_data = TiffImageLoader(self.tiff_file)
        self.raw_metadata = TiffMetadataLoader(self.tiff_file)
        
        
    def _load_ndata(self):
        """Eager loader for NDATA files"""
        with zipfile.ZipFile(self.file_path, 'r') as zip_file:
            # Loading the data
            with zip_file.open('data.npy') as f:
                data = np.load(f)
            
            # Loading metadata
            try:
                with zip_file.open('metadata.json') as f:
                    metadata = json.load(f)
            except KeyError:
                metadata = {}

        class ImmediateImageLoader:
            def __init__(self, data):
                self.data = data
                self.shape = data.shape
                
            def __getitem__(self, index):
                if self.data.ndim == 3:
                    return self.data[index]
                elif self.data.ndim == 2 and index == 0:
                    return self.data
                else:
                    raise IndexError("Index out of bounds for 2D data")
            def __len__(self):
                return self.data.shape[0] if self.data.ndim == 3 else 1
        
        class ImmediateMetadataLoader(MetadataMethods):
            def __init__(self, metadata, data):
                self.metadata = metadata
                self.data = data

            def __getitem__(self, index):
                return self.metadata

            def __len__(self):
                return 1
                
            def _get_full_metadata(self):   
                return self.metadata

        self.raw_data = ImmediateImageLoader(data)
        self.raw_metadata = ImmediateMetadataLoader(metadata,data)
    
    def _load_pkl(self):
        """Loader for PKL files"""
        data = joblib.load(self.file_path)
        
        if 'stack' in self.file_path:
            # Image stack pickle
            self.raw_data = data
            self.raw_metadata = {}
        elif 'metadata' in self.file_path:
            # Metadata pickle
            class PklMetadataLoader(MetadataMethods):
                def __init__(self, data):
                    self.data = data
                
                def __getitem__(self, index):
                    return self.data
                
                def __len__(self):
                    return 1
                
                def _get_full_metadata(self):
                    return self.data
                    
            self.raw_metadata = PklMetadataLoader(data)
            self.raw_data = np.empty(0)  # Empty placeholder
        else:
            raise ValueError("Unsupported pickle file content")
    
    def close(self):
        """Clean up resources"""
        if self.format == 'h5' and hasattr(self, 'h5_file'):
            self.h5_file.close()
        elif self.format == 'tiff' and hasattr(self, 'tiff_file'):
            self.tiff_file.close()
        elif self.format == 'ndata' and hasattr(self, 'zip_file'):
            self.zip_file.close()
    
    def __del__(self):
        """Destructor to ensure resources are freed"""
        self.close()


#================== The main DataLoader class for reading all data formats ==================
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.loader = UnifiedLoader(file_path)
        
        # Access data and metadata through unified interface
        self.raw_data = self.loader.raw_data
        self.raw_metadata = self.loader.raw_metadata

    def __getitem__(self, index):
        return self.raw_data[index]
    
    def __len__(self):
        return len(self.raw_data)
    
    def get_frame(self, index):
        return self[index]

    def get_metadata(self, target_key, data=None):
        return self.raw_metadata.get_metadata(target_key, data)
    
    def get_specific_metadata(self, target_key, required_keys=None, data=None, under_required_keys=False):
        return self.raw_metadata.get_specific_metadata(
            target_key, required_keys, data, under_required_keys
        )
    
    def get_metadata_item(self, key):
        """Get metadata item by key (for HDF5 metadata access)"""
        return self.raw_metadata[key]
  


#####################  EELS Lazy Loader #####################
class EELSLazyLoader(MetadataMethods):
    def __init__(self, h5_source, key=None):
        if isinstance(h5_source, str):
            self._h5_file = h5py.File(h5_source, 'r')
            self._needs_close = True
        else:
            self._h5_file = h5_source
            self._needs_close = False

        self._pair_keys = sorted(list(self._h5_file['pairs'].keys()))
        first_pair = self._h5_file['pairs'][self._pair_keys[0]]
        self._available_keys = list(first_pair.keys())

        # Create attributes for each available key
        for key_name in self._available_keys:
            setattr(self, key_name, self._create_lazy_accessor(key_name))

    def _create_lazy_accessor(self, key):
        class LazyAccessor:
            def __init__(self_inner, h5_file, pair_keys, key):
                self_inner._h5_file = h5_file
                self_inner._pair_keys = pair_keys
                self_inner.key = key

            def __getitem__(self_inner, index):
                group = self_inner._h5_file['pairs'][self_inner._pair_keys[index]]
                dataset = group[self_inner.key]
                if isinstance(dataset, h5py.Dataset) and dataset.dtype.kind in {'S', 'O'}:
                    return json.loads(dataset.asstr()[()])
                return dataset[:]

            def __len__(self_inner):
                return len(self_inner._pair_keys)
                
            def _get_full_metadata(self_inner):
                return [self_inner[i] for i in range(len(self_inner))]

        return LazyAccessor(self._h5_file, self._pair_keys, key)

    def __getitem__(self, index):
        group = self._h5_file['pairs'][self._pair_keys[index]]
        return {key: group[key][()] for key in self._available_keys}
    
    def __len__(self):
        return len(self._pair_keys)
    
    def _get_full_metadata(self):
        return [self[i] for i in range(len(self))]
                
    def close(self):
        if self._needs_close:
            self._h5_file.close()
            
    def get_original_filenames(self, index):
        """Get original filenames for a pair"""
        pair_group = self._h5_file['pairs'][self._pair_keys[index]]
        return {
            'point_spectrum': pair_group.attrs.get('point_spectrum_file', ''),
            'image': pair_group.attrs.get('image_file', '')
        }
    


    if __name__ == "__main__":
        # Example usage
        filepath_ndata = '/home/somar/Desktop/own_stuff/cleaning script/sample data files/081.ndata1'
        filepath_tiff = '/home/somar/Desktop/own_stuff/cleaning script/sample data files/VCR stack (MAADF) cluster_2 14.11.tif'
        filepath_pkl = '/home/somar/Desktop/own_stuff/cleaning script/sample data files/stack.pkl'
        metadata_pkl = '/home/somar/Desktop/own_stuff/cleaning script/sample data files/metadata.pkl'
        filepath_h5 = '/home/somar/Desktop/2025/Data for publication/Multilayer graphene/Sample 2476/stack.h5'
        filepath_eels = '/home/somar/Desktop/own_stuff/cleaning script/sample data files/eels_point_spectrum_pairs.h5'
        stacks_ssb = ['/home/somar/Desktop/2025/Data for publication/Sample 2525/SSB reconstruction of 4d STEM data/stack_ssbs.h5']
        file_path = '/home/somar/Desktop/own_stuff/cleaning script/sample data files/SuperScan-MAADF_2025-02-26T153932.717062_2048x2048_0.ndata1'
        img_seq = DataLoader(filepath_h5)
        data = img_seq.get_frame(22)
        data1 = img_seq.raw_data[44]
        print(data.shape, data1.shape)
        # print(data, data1)
