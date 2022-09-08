from keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from keras.applications.resnet import preprocess_input,decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'D:\study_data\_data\dog\*.jpg'
img = image.load_img(img_path, target_size=(224,224))
print(img)

x = image.img_to_array(img)
print('=================image.img_to_array(img)==========')
print(x, '\n',x.shape) # (224, 224, 3)
print(np.min(x),np.max(x)) # 0.0 255.0

x = np.expand_dims(x,axis=0)
print('=================expand_dims==========')
print(x, '\n',x.shape)
# #  axis = 0 ===> (1, 224, 224, 3)
# #  axis = 1 ===> (224, 1, 224, 3)
# #  axis = 2 ===> (224, 224, 1, 3)
x = preprocess_input(x) # 인풋 전처리 스케일링 해줌 
print('=================preprocess_input==========')
print(x, '\n',x.shape)
print(np.min(x),np.max(x)) # -119.68 151.061
print('=================predict==========')

preds = model.predict(x)
print(preds, '\n',preds.shape)

print("결과는 : ",decode_predictions(preds,top=5)[0])




