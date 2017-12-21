import tensorflow as tf
import numpy as np
import utils
import model
import scipy.misc
import argparse
from argument import add_arguments

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW4")
    parser = add_arguments(parser)
    args = parser.parse_args()
    return args

x_tags = utils.load_tags('./data/', preload=True)
 
# Define model
gan = model.GAN(parse())
_, _, _, _, _, _ = gan.build_model()
    
# Launch session
config = tf.ConfigProto(device_count={'GPU':0}) 
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './models/latest_model.ckpt')

input_tensors, outputs = gan.build_generator()
z_noise = np.random.normal([1, 100])
[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : x_tags[0],
				input_tensors['t_z'] : z_noise,
			} )

scipy.misc.imsave('./fake_image_0.jpg', gen_image[0])


	

