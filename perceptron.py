import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard as tb

def read_data_from_file(file_name):
    input_file = open(file_name, 'r', encoding='utf-8')

    X = []
    Y = []

    for line in input_file.readlines():

        cur_line = line.strip()
        row = cur_line.split()
        x = list(map(float, row[0:-1]))
        y = int(row[-1])
        if (y != 1): y = -1

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

def plot_data(X, Y):

    for i in range(X.shape[0]):
        if (Y[i] == 1):
            plt.plot(X[i][0], X[i][1], 'ro')
        else :
            plt.plot(X[i][0], X[i][1], 'bo')

    plt.show()


X_input, Y_input = read_data_from_file('A.txt')
#plot_data(X_input, Y_input)

#parameters
learning_rate = 0.1
training_epochs = 100
display_step = 1

n_hidden = 2
n_input = 2
n_classes = 2
n_output = 1

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, 1])

#weight
w1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
b1 = tf.Variable(tf.random_normal([n_hidden]))

w2 = tf.Variable(tf.random_normal([n_hidden, n_output]))
b2 = tf.Variable(tf.random_normal([n_output]))

def perceptron(x):
    #fc layer
    hidden_layer = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
    output_layer = tf.add(tf.matmul(hidden_layer, w2), b2)

    return output_layer

logits = perceptron(X)

#loss and optimizer
loss = tf.reduce_mean(tf.nn.relu(- Y * logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #training cycle
    for epoch in range(training_epochs):

        batch_size = X_input.shape[0]
        training_loss = 0

        for i in range(batch_size):
            sample_x, sample_y = X_input[i], Y_input[i]
            sample_x = sample_x.reshape((-1, n_input))
            sample_y = sample_y.reshape((-1, 1))
            _, sb, t = sess.run([train_op, logits, loss], feed_dict={X : sample_x, Y: sample_y})
            #print("logits = ", sb)
            #print("t = ", t)
            training_loss += t / batch_size

        if epoch % display_step == 0:
            print("Epoch ", '%d' % (epoch + 1), "training_loss={:.9f}".format(training_loss))
        if training_loss == 0 : break

    print("Training Finished !")

    #test model

    one = tf.ones_like(logits)
    pred = tf.where(logits > 0, one, -one)
    correct_pred = tf.equal(pred, Y)
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sample_x = X_input.reshape((-1, n_input))
    sample_y = Y_input.reshape((-1, 1))

    print("acc = ", acc.eval({X: sample_x, Y: sample_y}))
    #print("logits = ", logits.eval({X: sample_x, Y: sample_y}))
    #print("pred = ", pred.eval({X: sample_x, Y: sample_y}))

    # get value
    w_1, b_1, w_2, b_2 = sess.run([w1, b1, w2, b2])
    print("w1 = ", w_1)
    print("b1 = ", b_1)
    print("w2 = ", w_2)
    print("b2 = ", b_2)

