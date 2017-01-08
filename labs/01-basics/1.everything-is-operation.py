import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a + b

print(c)            # Tensor

print(sess.run(c))  # 5