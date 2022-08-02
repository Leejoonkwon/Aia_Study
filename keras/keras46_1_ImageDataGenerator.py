import numpy as np      
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pp
from sklearn import datasets

train_datagen = ImageDataGenerator(
    rescale=1./255,               # MinMax 스케일링과 같은 개념 
    horizontal_flip=True,         # 인풋을 무작위로 가로로 뒤집습니다.
    vertical_flip=True,           # 인풋을 무작위로 세로로 뒤집습니다.
    width_shift_range=0.1,        #부동소수점: < 1인 경우 전체 가로넓이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    height_shift_range=0.1,       #부동소수점: < 1인 경우 전체 세로높이에서의 비율, >= 1인 경우 픽셀의 개수입니다.
    rotation_range=5,             #정수. 무작위 회전의 각도 범위입니다
    zoom_range=1.2,               #부동소수점 혹은 [하한, 상산]. 무작위 줌의 범위입니다(이미지를 확대 및 축소)
    shear_range=0.7,              #부동소수점. 층밀리기의 강도입니다. 이미지를 찌그러 트린다.(회전과 다름)
    fill_mode='nearest' )         # {"constant", "nearest", "reflect" 혹은 "wrap"} 중 하나. 
                                  # 회전 축소 등으로 이미지에 여백이생겼을때 채우는 방법입니다.
# 디폴트 값은 'nearest'입니다. 인풋 경계의 바깥 공간은 다음의 모드에 따라 다르게 채워집니다:
#  cval: 부동소수점 혹은 정수. fill_mode = "constant"인 경우 경계 밖 공간에 사용하는 값입니다.
# 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
# 'nearest': aaaaaaaa|abcd|dddddddd
# 'reflect': abcddcba|abcd|dcbaabcd
# 'wrap': abcdabcd|abcd|abcdabcd
# 증폭 옵션 1장의 이미지로도 다양한 데이터 증폭 가능하지만  데이터가 너무 적을 경우 과적합 발생 가능성 있음

test_datagen = ImageDataGenerator(
    rescale=1./255
) # test data는 원형을 건드리지 않는다.훈련한 데이터와 비교하기 위해서 변형 X
xy_train = train_datagen.flow_from_directory(
    'd:/_data/image/brain/train/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
   #예측해야 모델의 클래스에 따름 
    #categorical : 2D one-hot 부호화된 라벨이 반환됨. binary : 1D 이진 라벨이 반환됨. sparse : 1D 정수 라벨이 반환됨.
    shuffle=True,) # 경로 및 폴더 설정
# Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
     target_size=(200,200), #target_size는 본인 자유 
    batch_size=5,
    class_mode='binary',
    shuffle=True,
    ) # Found 120 images belonging to 2 classes.
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(xy_train[0][0][i],cmap='gray')
plt.show() 
################
# print(xy_train[0].shape)  # 160장을 batch 5로 잘라서 32장이므로  최대인수 31이다 
# # (5, 150, 150, 3)  흑백또한 컬러 안에 있으므로 마지막 컬럼이 3이다. color_mode 에서 'graysclae'로 흑백을 지정해주지 않으면 기본값은은 칼라(3)다.
# print(xy_train[1].shape)  # 160장을 batch 5로 잘라서 32장이므로  최대인수 31이다 
# #xy로 묶었기 때문에 나온 값이 [0]의 값은 x값이고 [1]의 값은 y값이다.
# print(xy_train[31][0],xy_train[31][0].shape) #[0](5, 200, 200, 3) #[1] (5,)
# print(type(xy_train)) 
# print(type(xy_train[0])) # <class 'tuple'> 변경할 수 없는 list 형태의 데이터 구성 
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

