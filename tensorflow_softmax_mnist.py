import tensorflow as tf

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Variables
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

w_h = tf.Variable(tf.random_normal([784, 625], mean=0.0, stddev=0.05))
w_o = tf.Variable(tf.random_normal([625, 10], mean=0.0, stddev=0.05))
b_h = tf.Variable(tf.zeros([625]))
b_o = tf.Variable(tf.zeros([10]))

# Create the model
def model(X, w_h, b_h, w_o, b_o):
    h = tf.sigmoid(tf.matmul(X, w_h) + b_h)
    pyx = tf.nn.softmax(tf.matmul(h, w_o) + b_o)

    return pyx

y_hypo = model(x, w_h, b_h, w_o, b_o)

# Cost Function basic term
cross_entropy = -tf.reduce_sum(y_*tf.log(y_hypo))

# Regularization terms (weight decay)
L2_sqr = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
lambda_2 = 0.01

# the loss and accuracy
loss = cross_entropy + lambda_2 * L2_sqr
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_hypo,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print('Training...')
    for i in range(20001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x: batch_xs, y_: batch_ys})
        if i % 2000 == 0:
            train_accuracy = accuracy.eval({x: batch_xs, y_: batch_ys})
            print('  step, accurary = %6d: %6.3f' % (i, train_accuracy))

    # Test trained model
    print('accuracy = ', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))