
import numpy as np
def init_params(train_dim):
	'''
	输入：
	train_dim：样本特征数
	输出：
	w：初始化后的权重
	b：初始化后的偏置
	'''
	w = np.zeros((train_dim,1))
	b = 0
	return w,b

def linear_regress(X,y,w,b):
	'''
	输入：
	X：输入数据
	y：输出标签
	w：权重
	b：偏置
	输出：
	y_hat：预测值
	loss：预测值与真实值的均方误差
	dw：权重的一阶梯度
	db：偏置的一阶梯度
	'''
	num_train = X.shape[0]
	num_feature = X.shape[1]
	y_hat = np.dot(X,w) + b
	loss = np.sum((y_hat-y)**2)/num_train
	dw = np.dot(X.T,(y_hat-y))/num_train
	db = np.sum((y_hat-y))/num_train
	return y_hat, loss, dw, db	

def train(X, y, learning_rate=0.01, epochs=10000):
	'''
	输入：
	X：输入数据
	y：输出标签
	learning_rate：学习率
	epochs：迭代次数
	输出：
	loss_his：每一代的误差
	params：参数字典
	grads：优化后的梯度
	'''
	loss_his = []
	w, b = init_params(X.shape[1])
	for i in range(epochs):
		y_hat, loss, dw, db = linear_regress(X, y, w, b)
		w += -learning_rate*dw
		b += -learning_rate*db
		loss_his.append(loss)
	params = {'w':w, 'b':b}
	grads = {'dw':dw,'db':db}
	return loss_his, params, grads

def predict(X, params):
	'''
	输入：
	X：测试数据集
	params：模型训练参数
	输出：
	y_pre：预测值
	'''
	w = params['w']
	b = params['b']
	y_pre = np.dot(X, w) + b
	return y_pre

