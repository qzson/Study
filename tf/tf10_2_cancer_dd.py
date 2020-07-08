# 20-07-08_31
# 이진분류 cancer

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import numpy as np
	
if __name__=='__main__':
	#### 1. scikit-learn의 breast cancer dataset 불러오기
	'''
	< NOTE >
	scikit-learn에서 제공하는 breast cancer dataset은 간단한 binary classification dataset임
	'''
	print('>>> 1. Load the breast cancer dataset from sklearn module')
	BRCA = load_breast_cancer()
	data = BRCA.data
	target = BRCA.target.reshape(BRCA.target.shape[0],1)
	
	
	#### 2. parameter 설정하기
	'''
	< NOTE >
	본 예제의 NN은 Input, Hidden, Output 레이어를 각각 1개씩 갖음
		- Input 레이어는 30개의 뉴런을 갖음
		- Hidden 레이어는 10개의 뉴런을 갖음
		- Output 레이어는 1개의 뉴런을 갖음
	'''
	print('>>> 2. Set parameters')
	N_SAMPLE, N_FEATURE = BRCA.data.shape  # data.shape == (569,30)
	N_LABEL = len(np.unique(BRCA.target))
	S_INPUT = N_FEATURE
	S_HIDDEN = 10
	S_OUTPUT = 1
	MAX_EPOCH = 10
	LEARNINGRATE = 0.0025

	print('    number of samples  : %d' % N_SAMPLE)
	print('    number of features : %d' % N_FEATURE)
	print('    number of labels   : %d' % N_LABEL)
	print('    size of input layer  : %d' % S_INPUT)
	print('    size of hidden layer : %d' % S_HIDDEN)
	print('    size of output layer : %d' % S_OUTPUT)
	print('    maximum epoch : %d' % MAX_EPOCH)
	print('    learning rate : %.3f' % LEARNINGRATE)
	
	
	#### 3. training & test data 만들기
	print('>>> 3. Break up the dataset into non-overlapping training (75%) and testing')
	skf = StratifiedKFold(n_splits=4)
	## 4개의 fold 중 첫번째 fold만 이용
	train_index, test_index = next(iter(skf.split(data, target)))
	## data
	X_train = data[train_index]
	X_test = data[test_index]
	## target
	Y_train = target[train_index]
	Y_test = target[test_index]
	
	print('    number of samples for train: %d' % X_train.shape[0])
	print('    number of samples for test : %d' % X_test.shape[0])

	
	#### 4. Tensorflow를 이용하여 Neural network 생성하기
	print('>>> 4. Construct a neural network model using Tensorflow')
	with tf.device('/cpu:0'):
		## 1) Define Input & Actual data
		X = tf.placeholder(tf.float32, [None, S_INPUT], name="InputData")
		Y = tf.placeholder(tf.float32, [None, S_OUTPUT], name="ActualLabel")

		## 2) Define weight variables
		W_ih = tf.Variable(tf.truncated_normal([S_INPUT, S_HIDDEN], stddev=0.01), name="Weight_IH")
		B_ih = tf.Variable(tf.truncated_normal([S_HIDDEN], stddev=0.01), name="Bias_IH")

		W_ho = tf.Variable(tf.truncated_normal([S_HIDDEN, S_OUTPUT], stddev=0.01), name="Weight_HO")
		B_ho = tf.Variable(tf.truncated_normal([S_OUTPUT], stddev=0.01), name="Bias_HO")

		## 3) Inference process
		H = tf.sigmoid(tf.matmul(X, W_ih) + B_ih)
		Z = tf.matmul(H, W_ho) + B_ho

		## 4) Cost function & Optimization
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z, labels=Y))
		optimizer = tf.train.GradientDescentOptimizer(LEARNINGRATE).minimize(cost)

		## 5) Compute prediction accuracy
		pred_Y = tf.cast(tf.sigmoid(Z) > 0.5, tf.float32)
		correction = tf.equal(pred_Y, Y)  # bool
		accuracy = tf.reduce_mean(tf.cast(correction, tf.float32))
	
	
	#### 5. Neural network 훈련하기
	print(">>> 5. Start learning the model with %d epochs" % MAX_EPOCH)
	display_step = 1
	with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
		## 1) Initialize all variables
		tf.global_variables_initializer().run()
		
		## 2) Start learning the model
		for step in range(MAX_EPOCH):
			_, C = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train})	
			A = accuracy.eval({X:X_test, Y:Y_test})
		
			if step % display_step == 0:
				print("    Epoch: %03d\tCost[tr]=%.5f\tAccuracy[te]=%.4f" % (step, C, A))
			
		print(">>> Optimization Finish!")
		
		## 3) Evaluation the optimized model using test dataset
		print("    Accuracy[te]: %.3f" % accuracy.eval({X:X_test, Y:Y_test}))