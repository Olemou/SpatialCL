"""
DISCO.thermal_imagery

Contains data handling, preprocessing, and augmentation logic
for thermal and RGB modalities.
"""


from .augmentor import ThermalAugmentor, thermal_erase, thermal_contrast,brightness_contrast, elastic_transform

__all__ = ["ThermalAugmentor", "thermal_erase","thermal_contrast","brightness_contrast","elastic_transform"]
