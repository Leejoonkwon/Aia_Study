from keras_cv_attention_models import yolox
import numpy as np
from PIL import Image

image = Image.open('D:\study_data\_data/random.jpg',mode='r')
image = np.array(image)
print(image.shape)

model = yolox.YOLOXS(pretrained="coco")
preds = model(model.preprocess_input(image[:, :, ::-1])) # RGB -> BGR
bboxs, lables, confidences = model.decode_predictions(preds)[0]

# Show result
from keras_cv_attention_models.coco import data
data.show_image_with_bboxes(image, bboxs, lables, confidences, num_classes=80)
 



