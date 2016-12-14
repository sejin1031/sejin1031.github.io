#Tensorflow를 사용하기위해 import
import tensorflow as tf
#55000개의 학습데이터 mnist.train, 10000개의 테스트데이터 mnist.test 및 mnist.validation 세 부분으로 나눠져있다.
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model/ 
#심볼릭 변수를 사용하여 상호작용하는 작업을 기술함
x = tf.placeholder(tf.float32, [None, 784])
#Variable은 가중치와 편향값을 추가적으로 입력하게함
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# W의 형태가 [784,10]이므로 784차원의 이미지벡터를 곱하여 10차원 벡터의 증거를 만들어야 한다
# 이때 단순하게 출력에 [10]을 더하면된다.
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 교차엔트로피를 구현하기위해 우선적으로 정답을 입력하기 위한 새 placeholder를 추가한다
y_ = tf.placeholder(tf.float32, [None, 10])
# 그 다음 교차엔트로피를 구현할 수 있다.
# 첫번째로 tf.log는 y의 각 원소의 로그값을 계산한다
# 그 다음 y_의 각 원소에 해당하는 tf.log(y)를 곱한다
# 마지막으로 tf.reduce_sum은 텐서의 모든 원소를 더한다
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# 이 때 TensorFlow에게 학습도를 0.01로 준 경사하강법 알고리즘을 이용하여
# 교차엔트로피를 최소화하도록 명령
# 경사하강법은 TensorFlow가 각각의 변수들을 비용을 줄이는 방향으로 약간씩 바꾸는 간단한 방법
# TensorFlow가 실제로 하는 일은 역전파 및 경사 하강법을 구현한 작업을 그래프에 추가
# 실행할 경우 비용을 줄이기 위해 변수를 살짝 미세조정
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
# 실행 전 마지막으로 우리가 만든 변수들을 초기화하는 작업
init = tf.initialize_all_variables()
# 세션에서 모델을 시작하고 변수들을 초기화하는 작업
sess = tf.Session()
sess.run(init)

# Learning
# 이제 학습을 할겁니다 1000번 반복할테죠
for i in range(1000):
# 각 반복 단계마다 학습세트로부터 100개의 무작위 데이터들의
# 일괄처리들을 가져옵니다
# placeholders를 대체하기위한 일괄처리데이터에
# train_step 피딩을 실행
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
# tf.argmax는 특정한 축을 따라 가장 큰 원소의 색인을 알려주는 유용한 함수
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 이 결과는 부울리스트를 준다
# 부정소숫점으로 캐스팅한 후 평균값을 구하면 된다
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
# 테스트 데이터를 대상으로 정확도를 확인한다
# 몇몇 작은 변경만으로 97%를 얻을 수 있다
# 최고의 모델은 99.7%의 정확도가 있을 것이다
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
