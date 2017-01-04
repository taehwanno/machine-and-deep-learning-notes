import tensorflow as tf

x_data = [1, 2, 3]  # training data
y_data = [1, 2, 3]  # traning

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but Tensorflow will figure that out for us.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # Weight
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis
hypothesis = W * x_data + b  # H(x) = Wx + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))  # cost(W, b) = 1/m * (sigma(H(x_i) - y_i))^2

# Minimize (Block-box codes 18 ~ 20)
a = tf.Variable(0.1)  # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.global_variables_initializer()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# Learns best fit is W: [0.1], b: [0.3]
