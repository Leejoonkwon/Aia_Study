from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D #2d는 이미지 

model = Sequential()
# model.add(Dense(units=10,input_shape=(10,10,3))) #(batch_size, input_dim)
# model.summary()
# (input_dim + bias) * units = summary param 갯수 (Dense 모델,일반적 NN모델)

model.add(Conv2D(filters =10, kernel_size=(2,2),
                 input_shape=(10,10,2)))# (batch_size,row,column,channels)
model.add(Conv2D(7,(2,2),activation='relu')) #input_shape(9,9,10)
#filters와 kernel_size라고 쓰지않아도 해당 위치에
#  정수를 기입하는 경우 정상적으로 인식하고 작동한다.

# 파라미터는 kernel_size=(2,2) ((2 x 2)(커널 사이즈) x 2(채널) +1(bias)) x 10(아웃풋 노드) = 90(CNN  모델)
# Dense와 다르게 CNN에서는 그래픽 카드 성능이 허용하는 한 filters를 자유롭게 설정할 수 있다.(3090사라)
#kernel_size (x,y) x 와 y 의 사이즈로 잘라 데이터(이미지)를 검증한다.
# (Conv2D(10 -> 10개의 채널로 운영하겠다.
# kernel_size=(2,2) 필터는 (2,2)의 크기로 설정하겠다.
# input_shape=(5,5,1) 인풋되는 데이터는 가로 5행 세로 5행의 흑백의 데이터를 넣겠다.

# model.summary()

# Total params: 50
# 연산이 50번인 이유는 
# Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
# dilation=1, groups=1, bias=True, padding_mode='zeros')의 파라미터는 다음과 같습니다.

# in_channels: 입력 채널 수을 뜻합니다. 흑백 이미지일 경우 1,
# RGB 값을 가진 이미지일 경우 3 을 가진 경우가 많습니다.
# out_channels: 출력 채널 수을 뜻합니다.
# kernel_size: 커널 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다.
# stride: stride 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다. 기본 값은 1입니다.
# padding: padding 사이즈를 뜻합니다. int 혹은 tuple이 올 수 있습니다. 기본 값은 0입니다.
# padding_mode: padding mode를 설정할 수 있습니다. 기본 값은 'zeros' 입니다.
# 아직 zero padding만 지원 합니다.
# dilation: 커널 사이 간격 사이즈를 조절 합니다. 해당 링크를 확인 하세요.
# groups: 입력 층의 그룹 수을 설정하여 입력의 채널 수를 그룹 수에 맞게 분류 합니다. 
# 그 다음, 출력의 채널 수를 그룹 수에 맞게 분리하여, 입력 그룹과 출력 그룹의 짝을 지은 
# 다음 해당 그룹 안에서만 연산이 이루어지게 합니다.
# bias: bias 값을 설정 할 지, 말지를 결정합니다. 기본 값은 True 입니다.
# tf.keras.layers.Dense(
#     units,->아웃풋 노드의 개수
#     activation=None,-> 활성화 함수 지정
#     use_bias=True, ->y = wx+b 에서 b의 노드 투입 여부 결정
#     kernel_initializer="glorot_uniform",
#     bias_initializer="zeros",
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     **kwargs

# Input shape

# N-D tensor with shape: (batch_size, ..., input_dim).
#  The most common situation would be a 2D input with shape (batch_size, input_dim).
#########3