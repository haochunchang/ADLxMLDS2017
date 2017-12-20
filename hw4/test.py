import tensorflow as tf
import numpy as np
import utils
import model
import scipy.misc

x_tags = utils.load_tag('./data/', preload=True)
 
# Define model
gan = model.GAN(args)
_, _, _, _, _ = gan.build_model()
    
# Launch session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './models/')

input_tensors, outputs = gan.build_generator()
z_noise = np.random.normal([1, 100])
[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : x_tags[0],
				input_tensors['t_z'] : z_noise,
			} )

scipy.misc.imsave('./fake_image_0.jpg', gen_image[0])


	

