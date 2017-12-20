import argparse, random, shutil, os
from os.path import join
import scipy.misc
import tensorflow as tf
import numpy as np
import pandas as pd

import utils
import model
from argument import add_arguments

def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW4")
    parser = add_arguments(parser)
    args = parser.parse_args()
    return args
    
def train(args):
    random.seed(9487)
    np.random.seed(9487)
    # Load data
    x_tags = utils.load_tags(args.data_dir, preload=bool(args.preload))
    x_imgs = utils.load_data(args.data_dir, preload=bool(args.preload))
    loaded_data = {'tags': x_tags, 'images': x_imgs}
  
    size = x_imgs.shape[0]
    num_update_d = 5
    num_update_g = 1
    config = tf.ConfigProto(device_count={'GPU':1})
    output_path = './outputs'
    model_path = './models/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(model_path):
        os.makedirs('./models')
    
    # Define model
    gan = model.GAN(args)
    input_tensors, variables, loss, outputs, checks, optims = gan.build_model()
    
    # Launch session
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver(max_to_keep=2)
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    index = np.arange(size)
    d_loss_his = []
    # Start Training Algorithm
    for i in range(args.epochs):
        batch_no = 0
        np.random.shuffle(index)
        while batch_no*args.bz < size:
            real_images, wrong_images, caption_vectors, z_noise = get_batch(index, batch_no, args.bz, loaded_data)
            
            for i in range(num_update_d):
                # update discriminator
                check_ts = [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3']]
                _, d_loss, gen, d1, d2, d3 = sess.run([optims['d_optim'], loss['d_loss'], outputs['generator']] + check_ts,
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_wrong_image'] : wrong_images,
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['t_z'] : z_noise
                    })
                d_loss_his.append(d_loss)
            print("Last {} average loss of D: {}".format(num_update_d, np.mean(d_loss_his[-5:])))
            for i in range(num_update_g):
                # update generator
                _, g_loss, gen = sess.run([optims['g_optim'], loss['g_loss'], outputs['generator']],
                    feed_dict = {
                        input_tensors['t_real_image'] : real_images,
                        input_tensors['t_wrong_image'] : wrong_images,
                        input_tensors['t_real_caption'] : caption_vectors,
                        input_tensors['t_z'] : z_noise
                    })
            
            batch_no += 1
            if (batch_no % args.save_every) == 0:
                print("d_loss:{}, g_loss:{}, batch:{}, epochs:{}\n".format(d_loss, g_loss, batch_no, i))
                print("Saving Images, Model")
                save_for_vis(output_path, real_images, gen)
                save_path = saver.save(sess, os.path.join(model_path, "latest_model.ckpt")) 
        if i%10 == 0:
            save_path = saver.save(sess, os.path.join(model_path, "model_after_epoch_{}.ckpt".format(i)))

def get_batch(index, batch_no, batch_size, loaded_data):

    total_size = loaded_data['tags'].shape[0]
    if (batch_no+1)*batch_size > total_size:
        batch_end = total_size
        image_ids = index[batch_end-batch_size:batch_end]
    else:
        batch_end = (batch_no+1)*batch_size
        image_ids = index[batch_size*batch_no:batch_end]

    real_images = loaded_data['images'][image_ids, :, :, :]
    caption_vectors = loaded_data['tags'][image_ids, :]

    wrong_ids = np.random.choice(index, size=batch_size)
    wrong_images = loaded_data['images'][wrong_ids, :, :, :]
        
    z_noise = np.random.normal(size=[batch_size, 100])
    return real_images, wrong_images, caption_vectors, z_noise

def save_for_vis(data_dir, real_images, generated_images):
    
    if not os.path.exists( join(data_dir, 'samples')):
        os.makedirs( join(data_dir, 'samples') )

    for i in range(0, 10):
        real_image_255 = np.zeros( (96,96,3), dtype=np.uint8)
        real_images_255 = (real_images[i,:,:,:])
        scipy.misc.imsave( join(data_dir, 'samples/{}.jpg'.format(i)) , real_images_255)

        fake_image_255 = np.zeros( (96,96,3), dtype=np.uint8)
        fake_images_255 = (generated_images[i,:,:,:])
        scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)

if __name__ == '__main__':
    args = parse()
    train(args)