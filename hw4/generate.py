import tensorflow as tf
import numpy as np
import argparse, random, os
from os.path import join
import h5py
from skimage.io import imsave
from skimage.transform import resize

import model
from argument import add_arguments

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default='./models/latest_model.ckpt',
                       help='Trained Model Path')

    parser.add_argument('--n_images', type=int, default=5, help='Number of Images per Caption')

    parser.add_argument('--captions', type=str, default='sample_caption_vectors.hdf5',
                       help='Caption Thought Vector File')
    parser = add_arguments(parser)
    args = parser.parse_args()
    
    random.seed(9487)
    np.random.seed(9487)
    txt_dim = 2400 
    model_options = {
        'z_dim' : 100,
        't_dim' : 256,
        'image_size' : 96,
        'gf_dim' : 64,
        'df_dim' : 64,
        'gfc_dim' : 1024,
        'caption_vector_length' : txt_dim
    }
    args.bz = args.n_images

    gan = model.GAN(args)
    _, _, _, _, _, _ = gan.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)
    
    input_tensors, outputs = gan.build_generator()

    h = h5py.File( args.captions )
    caption_vectors = np.array(h['vectors'])
    caption_image_dic = {}
    for cn, caption_vector in enumerate(caption_vectors):

        caption_images = []
        z_noise = np.random.normal(size=[args.n_images, model_options['z_dim']])
        caption = [ caption_vector[0:model_options['caption_vector_length']] ] * args.n_images
        
        [ gen_image ] = sess.run( [ outputs['generator'] ], 
            feed_dict = {
                input_tensors['t_real_caption'] : caption,
                input_tensors['t_z'] : z_noise,
            } )
        
        caption_images = [gen_image[i,:,:,:] for i in range(0, args.n_images)]
        caption_image_dic[ cn ] = caption_images
        print("Generated: {}".format(cn))
    
    output_path = './samples'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        for the_file in os.listdir(output_path):
            file_path = os.path.join(output_path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    for cn in range(0, len(caption_vectors)):
        caption_images = []
        for i, im in enumerate( caption_image_dic[ cn ] ):
            im_name = "sample_{}_{}.jpg".format(cn, i)
            im = resize(im, (64,64))
            imsave(join(output_path, im_name), im)

if __name__ == '__main__':
    main()
