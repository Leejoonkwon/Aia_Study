import autokeras as  ak
print(ak.__version__) # 1.0.20
import tensorflow as tf
import time
(x_train,y_train),(x_test,y_test) = \
    tf.keras.datasets.cifa100.load_data()
print(x_train.shape)

#2. 모델

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)
#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train, epochs=5)
end_time = time.time()-start_time
#4. 평가, 예측

y_predict = model.predict(x_test)

result = model.evaluate(x_test,y_test)
print('결과 :', result)
print('걸린 시간 :',round(end_time, 4))
