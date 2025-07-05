# Add this to voila_app.py
import os
from IPython.display import display
from ipywidgets import Button, FileUpload, Output, VBox
from imageanalysis.calibrated_images_class import CalibratedImages
import os
from traitlets.config import Config

# Configure Voila to allow all files
c = Config()
c.VoilaConfiguration.file_allowlist = ['.*']  # Allow all files
c.VoilaConfiguration.enable_nbextensions = True
stacks_ssb1 = 'stack.h5'
font_path = "SourceSansPro-Semibold.otf" 

class ContentManager:
    def __init__(self):
        self.out = Output()
        self.upload = FileUpload(description="Upload Files", multiple=True)
        self.button = Button(description="Process")
        self.button.on_click(self.process_files)
        
    def display(self):
        display(VBox([self.upload, self.button, self.out]))
        
    def process_files(self, b):
        for name, content in self.upload.value.items():
            with open(name, 'wb') as f:
                f.write(content['content'])
        self.out.clear_output()
        with self.out:
            print(f"Uploaded {len(self.upload.value)} files")
            # Initialize your app here
            calibrated_app = CalibratedImages(stacks_ssb1[0], font_path=font_path)
            display(calibrated_app.tabs)

# At the end of voila_app.py
if __name__ == "__main__":
    manager = ContentManager()
    manager.display()