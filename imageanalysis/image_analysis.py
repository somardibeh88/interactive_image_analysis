########
# Title: Interactive Image Analysis
# Author: Somar Dibeh
# Date: 2025-04-25
########

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
import matplotlib
matplotlib.use("Qt5Agg")
import os
import json
import h5py
import cv2
import csv
import joblib
import numpy as np
import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.offsetbox import AnchoredText
from matplotlib.font_manager import FontProperties 
import matplotlib.patches as patches
from datetime import datetime
from IPython.display import display, clear_output
from traitlets import link
import time
from ipywidgets import interactive_output, HBox, VBox, FloatSlider, IntSlider, Checkbox, Button, Output, Dropdown, IntText, FloatText, Text
from data_loader import DataLoader
from filters import *
from utils import *
from sklearn.cluster import DBSCAN
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
import imageanalysis.fft_calibration_class as fc
from imageanalysis.fft_calibration_class import *


class InteractiveImageAnalysis(QWidget):
    pass




