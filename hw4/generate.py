import tensorflow as tf
import numpy as np
import argparse, random, os
from os.path import join
import h5py
import scipy.misc

from Utils import image_processing
import model

def main():
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--model_path', type=str, default='./models/.ckpt',
                       help='Trained Model Path')

	parser.add_argument('--n_images', type=int, default=5, help='Number of Images per Caption')

	parser.add_argument('--caption_thought_vectors', type=str, default='Data/sample_caption_vectors.hdf5',
                       help='Caption Thought Vector File')

	args = parser.parse_args()
        
    txt_dim = 17599 
	model_options = {
		'z_dim' : 100,
		't_dim' : 256,
		'batch_size' : args.n_images,
		'image_size' : 96,
		'gf_dim' : 64,
		'df_dim' : 64,
		'gfc_dim' : 1024,
		'caption_vector_length' : txt_dim
	}

	gan = model.GAN(args)
	_, _, _, _, _ = gan.build_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(sess, args.model_path)
	
	input_tensors, outputs = gan.build_generator()

	h = h5py.File( args.caption_thought_vectors )
	caption_vectors = np.array(h['vectors'])
	caption_image_dic = {}
	for cn, caption_vector in enumerate(caption_vectors):

		caption_images = []
		z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])
		caption = [ caption_vector[0:args.caption_vector_length] ] * args.n_images
		
		[ gen_image ] = sess.run( [ outputs['generator'] ], 
			feed_dict = {
				input_tensors['t_real_caption'] : caption,
				input_tensors['t_z'] : z_noise,
			} )
		
		caption_images = [gen_image[i,:,:,:] for i in range(0, args.n_images)]
		caption_image_dic[ cn ] = caption_images
		print "Generated", cn

	for f in os.listdir( join(args.data_dir, 'val_samples')):
		if os.path.isfile(f):
			os.unlink(join(args.data_dir, 'val_samples/' + f))

	for cn in range(0, len(caption_vectors)):
		caption_images = []
		for i, im in enumerate( caption_image_dic[ cn ] ):
			# im_name = "caption_{}_{}.jpg".format(cn, i)
			# scipy.misc.imsave( join(args.data_dir, 'val_samples/{}'.format(im_name)) , im)
			caption_images.append( im )
			caption_images.append( np.zeros((64, 5, 3)) )
		combined_image = np.concatenate( caption_images[0:-1], axis = 1 )
		scipy.misc.imsave( join(args.data_dir, 'val_samples/combined_image_{}.jpg'.format(cn)) , combined_image)


if __name__ == '__main__':
	main()
