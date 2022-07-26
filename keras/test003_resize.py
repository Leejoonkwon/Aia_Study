
import os
import glob
from PIL import Image

files = glob.glob('D:/project/감동/*')

for f in files:
    try:
        img = Image.open(f)
        img_resize = img.resize((150, 150), Image.LANCZOS)
        title, ext = os.path.splitext(f)
        img_resize.save(title + '_half' + ext)
    except OSError as e:
        pass    
# save 위치 바꿔 보기    