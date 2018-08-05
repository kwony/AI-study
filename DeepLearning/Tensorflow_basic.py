# # 1. Learn Tensorflow Basic

import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')
print(hello)

a = tf.constant(10)
b = tf.constant(32)

c = tf.add(a, b)
print(c)

sess = tf.Session()
sess.run(c)

result = sess.run(hello)
print(result)
print(type(result))
print(result.decode(encoding='utf-8'))
sess.close()

X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1,2,3],[4,5,6]]

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('=== x_data ===')
print(x_data)
print('=== W ===')
print(sess.run(W))
print('=== b ===')
print(sess.run(b))
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train_op = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
    _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, Y: y_data})  
    print(step, cost_val, sess.run(W), sess.run(b))

print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
print("X: 2.5, y:", sess.run(hypothesis, feed_dict={X: 2.5}))
