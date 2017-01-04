import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)

with tf.Session() as sess:
    print("Addition with variables: ", sess.run(add, feed_dict={a: 2, b: 3}))           # 5
    print("Multiplication with variables: ", sess.run(mul, feed_dict={a: 2, b: 3}))     # 6
