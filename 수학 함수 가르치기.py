import numpy as np
from deep_convnet import DeepConvNet
from common.trainer import Trainer
from modules.plot import *
import time as t

def f(x): return 2*x+1
train_nums, test_nums = range(-100, 101), range(-10000, 10001, 100)

x_train, t_train = [[x] for x in train_nums], [[f(x)] for x in train_nums]
x_train, t_train = np.array(x_train, dtype=float), np.array(t_train, dtype=float)
x_test, t_test = [[x] for x in test_nums], [[f(x)] for x in test_nums]
x_test, t_test2 = np.array(x_test, dtype=float), np.array(t_test, dtype=float)

network = DeepConvNet(params = [1, 10, 1], layers_info = [
        'affine', 'Relu',
        'affine', 'squaredloss'])
network.load_params('0000000259.pkl')
# network.save_params('params_init.pkl')

# # 학습하기
# start = t.time()
# trainer = Trainer(network, x_train, t_train, x_test, t_test, epochs=10000,
#                   mini_batch_size=len(train_nums), verbose=False,
#                   optimizer='adagrad', optimizer_param={'lr':1e-1},
#                   give_up={'epoch':2000, 'test_loss':1000}) # 가중치 초깃값 중요!
# trainer.train()
# print("\n%.4f sec\n" % (t.time() - start))

# # 학습 결과 출력
# network.save_params()
# train_loss, test_loss = network.loss(x_train, t_train), network.loss(x_test, t_test)
# name = str(format(test_loss, ".17f"))[-17:]
# print("new:", format(train_loss, ".10f"), format(test_loss, ".17f"))

network.load_params('0000000259.pkl')
train_loss1, test_loss1 = network.loss(x_train, t_train), network.loss(x_test, t_test)
print("old:", format(train_loss1, ".10f"), format(test_loss1, ".17f")) # print(network.params)

# if test_loss < 0.010000:
    # network.load_params('params_init.pkl')
    # network.save_params(f'{name}_init.pkl')

    # # 학습 진행도 그래프 그리기
    # loss_list = {"train_loss": trainer.train_loss_list, "test_loss":trainer.test_loss_list}
    # plot_loss_graphs(loss_list, ('train_loss', 'test_loss'), smooth=True, ylim=0.01)
    # plt.show()

# if test_loss < test_loss1:
    # network.load_params('params.pkl')
    # network.save_params(f'{name}.pkl')
    
# 문제 풀어보기
x_Quiz = [-100000000, -10000000, -1000000, -100000, -10000, -10.5,
          10000, 100000, 1000000, 10000000, 100000000]
answer = network.predict(np.array(x_Quiz))
for Q, a in zip(x_Quiz, answer): print(format(a[0], ".4f"), f"(Q:{Q})")