import tensorflow as tf
import numpy as np
import argparse, random, os
from os.path import join
from skimage.io import imsave
from skimage.transform import resize
import h5py

import model, utils
from argument import add_arguments

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default='./models/latest_model.ckpt',
                       help='Trained Model Path')

    parser.add_argument('--n_images', type=int, default=5, help='Number of Images per Caption')

    parser.add_argument('--captions', type=str, default='./data/sample_testing_text.txt',
                       help='Caption Thought Vector File')
    parser = add_arguments(parser)
    args = parser.parse_args()
    return args

def main(args):
    
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

    h = h5py.File('./tag_mapping.hdf5')
    captions = []
    with open(args.captions, 'r') as f:
        for line in f:
            captions.append(line.split(',')[-1].strip())

    caption_vectors = []
    for c in captions:
        caption_vectors.append(h.get(c))

    caption_vectors = np.array(caption_vectors)
    caption_vectors = np.reshape(caption_vectors, (caption_vectors.shape[0], txt_dim))
    print(caption_vectors.shape)
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
        #print("Generated: {}".format(cn))
    
    output_path = './samples'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        for the_file in os.listdir(output_path):
            file_path = os.path.join(output_path, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    #print(caption_vectors.shape)
    for cn in range(0, len(caption_vectors)):
        caption_images = []
        for i, im in enumerate( caption_image_dic[ cn ] ):
            im_name = "sample_{}_{}.jpg".format(cn, i+1)
            im = resize(im, (64,64))
            imsave(join(output_path, im_name), im)

if __name__ == '__main__':
    args = parse()
    main(args)
