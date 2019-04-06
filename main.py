#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Loading VGG
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    # Getting VGG layers
    graph = tf.get_default_graph()
    img_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
        
    return img_input, keep, layer_3, layer_4, layer_7
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # Replacing FC layer by a 1D convolution layer to preserve spatial information
    # Regularization helps penalize large weight values during training
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Reverse convolution and stride of 2 helps upsample
    fcn_1 = tf.layers.conv2d_transpose(conv_1x1, filters=vgg_layer4_out.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Skip connection
    sc_1 = tf.add(fcn_1, vgg_layer4_out)
    
    # Reverse convolution and stride of 2 helps upsample
    fcn_2 = tf.layers.conv2d_transpose(sc_1, filters=vgg_layer3_out.get_shape().as_list()[-1], kernel_size=4, strides=(2,2), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # Skip connection
    sc_2 = tf.add(fcn_2, vgg_layer3_out)
    
    # Reverse convolution and stride of 2 helps upsample
    fcn_3 = tf.layers.conv2d_transpose(sc_2, num_classes, 16, strides=(8,8), padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))    
    
    return fcn_3
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    
    # Reshape 4D tensors into 2D, one for each class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_lbl_reshaped = tf.reshape(correct_label, (-1, num_classes))
    
    # Standard cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_lbl_reshaped))
    
    # Training options
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_prob_val = 0.5
    learning_rate_val = 0.001

    for epoch in range(epochs):
        total_loss = 0
        idx = 0
        print("Epoch {} ---".format(epoch + 1))
        for img_batch, lbl_batch in get_batches_fn(batch_size):
            loss, _ = sess.run([cross_entropy_loss, train_op], feed_dict={input_image: img_batch, correct_label: lbl_batch, keep_prob: keep_prob_val, learning_rate: learning_rate_val})
            total_loss += loss
            idx = idx + 1
            print("Iteration {}".format(idx))
            print("Loss = {:.3f}".format(total_loss))
            
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = '/data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    # Training parameters
    epochs = 40
    batch_size = 16
    # keep_prob = 0.5
    # learning_rate = 0.001
    
    # Placeholders
    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, layer_3, layer_4, layer_7 = load_vgg(sess, vgg_path)
        final_layer = layers(layer_3, layer_4, layer_7, num_classes)
        logits, train_op, loss = optimize(final_layer, correct_label, learning_rate, num_classes)
        
        # Initialize
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        print("---Starting training---")

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, input_image, correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        
        print("---Done---")

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
