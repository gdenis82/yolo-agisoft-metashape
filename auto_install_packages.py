# ## For Agisoft Metashape Professional 2.3.0
# - python 3.12
#
# #### Based on:
# - https://github.com/agisoft-llc/metashape-scripts/blob/master/src/detect_objects.py
# - https://docs.ultralytics.com/
#
# ## How to install (Windows):
# How to install external Python module to Metashape Professional package https://agisoft.freshdesk.com/support/solutions/articles/31000136860-how-to-install-external-python-module-to-metashape-professional-package



import os, pathlib

import Metashape

from modules.pip_auto_install import pip_install, user_packages_location, _is_already_installed

# Checking compatibility
compatible_major_version = "2.3"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))

pathlib.Path(user_packages_location).mkdir(parents=True, exist_ok=True)
temporary_file = os.path.join(user_packages_location, "temp_links.html")

requirements_txt = """-f "{find_links_file_path}"
--index-url https://pypi.org/simple
--extra-index-url https://download.pytorch.org/whl/cu128
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128

deepforest==2.0.0
pytorch-lightning==2.6.1
albumentations==2.0.8

rasterio==1.4.3

shapely==2.0.7
Rtree
tqdm
ultralytics
scikit-learn==1.6.1""".format(find_links_file_path=temporary_file)

pip_install(requirements_txt)