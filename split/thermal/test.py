from thermal import  thermal_erase, thermal_contrast, brightness_contrast, elastic_transform  

from pathlib import Path
from PIL import Image


# project_root = parent of DISCO
project_root = Path(__file__).resolve().parent.parent  # adjust depending on where script is
img_path = project_root / "assets" / "test.png"
print(img_path)
img_path = "C:/Users/olemo/Documents/UCL-VAR/assets/test.png"
img = Image.open(img_path)
img = img.convert("RGB")
new = elastic_transform(img)
new.show()





