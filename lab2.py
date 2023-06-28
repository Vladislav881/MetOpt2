import math

import numpy as np

from random import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import psutil as psutil
import time

start = time.time()

count = 200
step = 0.25


def linear_regression_func(x, b, k):
    return b + k * x


def func(x):
    return 3 + 0.5 * x


def generate_sample(total):
    x = 0
    while x < total * step:
        yield func(x) + np.random.uniform(-0.2, 0.2) * np.random.uniform(2, 8)
        x += step


# сгенели все точки для прямой
X = np.arange(0, count * step, step)
Y = np.array([round(y, 2) for y in generate_sample(count)])


Y_real = np.array([func(x) for x in X])

plt.scatter(X, Y, alpha=0.4)
plt.plot(X, Y_real, 'g', linewidth=2.0)

k_start = -5
b_start = 10

w_start = np.array([-5, 10])
alpha = 0.0002


def get_value(x, y, b, k):
    return [(y[i] - (b + x[i] * k)) ** 2 for i in range(len(x))]

def get_sum(x, y, b, k):
    return sum(get_value(x, y, b, k))


def get_grad(x, y, w, size):
    pows = np.array([x ** i for i in range(size)])
    return np.array([-2 * (y - np.sum(w * pows)) * pows[i] for i in range(size)])


def shuffle(permutation):
    return list(np.random.permutation(permutation))


def const_lr(iteration):
    return 0.0002


def stage_lr(iteration):
    return 0.01 * (0.9 ** ((iteration + 10) // 20))


def exp_lr(iteration):
    return math.exp(-0.05 * (iteration + 100))


def sth_gradient(xs, ys, w, lr, batch_size, max_iters):
    cur_w = w
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])

    iters = 0
    pos = 0
    while iters < max_iters:
        grad = np.array([0 for i in range(len(w))])
        if pos + batch_size - 1 >= cnt_dots:
            pos = 0
            perm = shuffle(perm)
        for j in range(pos, pos + batch_size):
            grad = grad + get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
        grad /= batch_size
        pos += batch_size
        cur_w = cur_w - lr(iters) * grad
        iters += 1
    return [cur_w, iters]


def momentum(cur_w, cur_momentum, grad, gamma, lr):
    next_momentum = gamma * cur_momentum + (1 - gamma) * grad
    next_w = cur_w - lr * cur_momentum
    return [next_w, next_momentum]


def momentum_gradient(xs, ys, w, lr, batch_size, max_iters):
    cur_w = w
    cur_momentum = np.array([0.0 for i in range(len(w))])
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])

    iters = 0
    pos = 0
    while iters < max_iters:
        grad = np.array([0.0 for i in range(len(w))])
        if pos + batch_size - 1 >= cnt_dots:
            pos = 0
            perm = shuffle(perm)
        for j in range(pos, pos + batch_size):
            grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
        grad /= batch_size
        pos += batch_size
        cur_w, cur_momentum = momentum(cur_w, cur_momentum, grad, 0.9, lr(iters))
        iters += 1
    return [cur_w, iters]


def nesterov(cur_w, cur_momentum, grad, gamma, lr):
    next_momentum = gamma * cur_momentum + (1 - gamma) * grad
    next_w = cur_w - lr * cur_momentum
    return [next_w, next_momentum]


def nesterov_gradient(xs, ys, w, lr, batch_size, max_iters):
    cur_w = w
    cur_nesterov = np.array([0.0 for i in range(len(w))])
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])

    iters = 0
    pos = 0
    while iters < max_iters:
        grad = np.array([0.0 for i in range(len(w))])
        if pos + batch_size - 1 >= cnt_dots:
            pos = 0
            perm = shuffle(perm)
        for j in range(pos, pos + batch_size):
            grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w - lr(iters) * 0.9 * cur_nesterov, len(cur_w))
        grad /= batch_size
        pos += batch_size
        cur_w, cur_nesterov = nesterov(cur_w, cur_nesterov, grad, 0.9, lr(iters))
        iters += 1
    return [cur_w, iters]


def ada_grad(cur_w, cur_grad, grad, lr):
    lr = 0.1
    new_grad = cur_grad + np.square(-grad)
    new_w = cur_w + lr * np.divide(grad, np.sqrt(new_grad))
    return [new_w, new_grad]


def ada_grad_gradient(xs, ys, w, lr, batch_size, max_iters):
    cur_w = w
    cur_ada_grad = np.array([0.0 for i in range(len(w))])
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])

    iters = 0
    pos = 0
    while iters < max_iters:
        for i in range(cnt_dots // batch_size):
            grad = np.array([0.0 for i in range(len(w))])
            if pos + batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + batch_size):
                grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
            grad /= batch_size
            pos += batch_size
            cur_w, cur_ada_grad = ada_grad(cur_w, cur_ada_grad, -grad, lr(iters))
        iters += 1
    return [cur_w, iters]


def rms_prop(cur_w, cur_grad, grad, gamma, lr):
    lr = 0.1
    new_grad = gamma * cur_grad + (1 - gamma) * np.square(-grad)
    new_w = cur_w + lr * np.divide(grad, np.sqrt(new_grad) + 0.0000001)
    return [new_w, new_grad]


def rms_prop_gradient(xs, ys, w, lr, batch_size, max_iters):
    cur_w = w
    cur_prop_grad = np.array([0.0 for i in range(len(w))])
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])

    iters = 0
    pos = 0
    while iters < max_iters:
        for i in range(cnt_dots // batch_size):
            grad = np.array([0.0 for i in range(len(w))])
            if pos + batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + batch_size):
                grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
            grad /= batch_size
            pos += batch_size
            cur_w, cur_prop_grad = rms_prop(cur_w, cur_prop_grad, -grad, 0.9, lr(iters))
        print(iters, ':', cur_w)
        iters += 1
    return [cur_w, iters]


def adam(cur_w, cur_momentum, cur_grad, grad, beta, gamma, lr):
    lr = 0.1
    new_momentum = beta * cur_momentum - (1 - beta) * grad
    new_grad = gamma * cur_grad + (1 - gamma) * np.square(-grad)
    #print(new_grad)
    new_w = cur_w - lr * np.divide(new_momentum, np.sqrt(new_grad) + 0.0000001)
    return [new_w, new_momentum, new_grad]


def adam_gradient(xs, ys, w, lr, batch_size, max_iters):
    cur_w = w
    cur_prop_grad = np.array([0.0 for i in range(len(w))])
    cur_momentum = np.array([0.0 for i in range(len(w))])
    cnt_dots = len(xs)
    perm = shuffle([i for i in range(cnt_dots)])

    iters = 0
    pos = 0
    while iters < max_iters:
        for i in range(cnt_dots // batch_size):
            grad = np.array([0.0 for i in range(len(w))])
            if pos + batch_size - 1 >= cnt_dots:
                pos = 0
                perm = shuffle(perm)
            for j in range(pos, pos + batch_size):
                grad += get_grad(xs[perm[j]], ys[perm[j]], cur_w, len(cur_w))
            grad /= batch_size
            pos += batch_size
            cur_w, cur_momentum, cur_prop_grad = adam(cur_w, cur_momentum, cur_prop_grad, -grad, 0.9, 0.9, lr(iters))
        iters += 1
    return [cur_w, iters]

#test1 = sth_gradient(list(X), list(Y), w_start, const_lr, len(list(X)), 10 ** 5)
#print(test1[0][0], '+', str(test1[0][1]) + " * x", test1[1])
#
# test2 = sth_gradient(list(X), list(Y), w_start, stage_lr, 1, 10 ** 5)
# print(test2[0][0], '+', str(test2[0][1]) + " * x", test2[1])
#
# test3 = sth_gradient(list(X), list(Y), w_start, exp_lr, 1, 10 ** 5)
# print(test3[0][0], '+', str(test3[0][1]) + " * x", test3[1])
#test_momentum = momentum_gradient(list(X), list(Y), w_start, const_lr, 1, 10 ** 5)
#print(test_momentum[0][0], '+', str(test_momentum[0][1]) + " * x", test_momentum[1])

#test_nesterov = nesterov_gradient(list(X), list(Y), w_start, const_lr, 1, 10 ** 5)
#print(test_nesterov[0][0], '+', str(test_nesterov[0][1]) + " * x", test_nesterov[1])

#test_ada_grad = ada_grad_gradient(list(X), list(Y), w_start, const_lr, 1, 10 ** 4)
#print(test_ada_grad[0][0], '+', str(test_ada_grad[0][1]) + " * x", test_ada_grad[1])

test_prop_grad = rms_prop_gradient(list(X), list(Y), w_start, const_lr, 1, 10 ** 4)
print(test_prop_grad[0][0], '+', str(test_prop_grad[0][1]) + " * x", test_prop_grad[1])



Y_current = np.array([linear_regression_func(x, test_prop_grad[0][0], test_prop_grad[0][1]) for x in X])

plt.plot(X, Y_current, 'b', linewidth=2.0)


print(psutil.virtual_memory())
end = time.time()
print("время ", end - start)

plt.show()
