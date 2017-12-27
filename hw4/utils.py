## Preprocess functions
import os, pickle, glob
import pandas as pd
import numpy as np
from skimage import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from skip_thoughts import configuration
from skip_thoughts import encoder_manager

def load_data(path, preload=False):
    if not preload:
        img_path = os.path.join(path, 'faces')
        img_all_path = sorted(glob.glob(os.path.join(img_path, '*.jpg')), 
                                key=lambda x: int(x.split('/')[-1].split('.')[0]))
        imgs = io.imread_collection(img_all_path)
        imgs = io.concatenate_images(imgs)
        imgs = imgs / 255.0
        print("Image matrix shape:{}".format(imgs.shape))
        np.save('img_matrix', imgs)
    else:
        imgs = np.load('img_matrix.npy')
        print("Loading true img matrix...shape:{}".format(imgs.shape))
    return imgs

def load_data64(path, clean_lst, preload=False):
    if not preload:
        img_path = os.path.join(path, 'faces')
        img_all_path = sorted(glob.glob(os.path.join(img_path, '*.jpg')), 
                                key=lambda x: int(x.split('/')[-1].split('.')[0]))
        img_all_path = img_all_path[clean_lst]
        imgcol = io.ImageCollection(img_all_path, load_func=lambda x: io.imread(x).resize((64,64)) / 255.0)
        imgs = imgcol.concatenate()
        print("Image matrix shape:{}".format(imgs.shape))
        np.save('img64_matrix', imgs)
    else:
        imgs = np.load('img64_matrix.npy')
        print("Loading true img matrix...shape:{}".format(imgs.shape))
    return imgs

def load_tags_clean(path, preload=False):
    if not preload:
        tags = pd.read_csv(os.path.join(path, 'tags_clean.csv'), header=None, index_col=0)
        tags['tags'] = tags.iloc[:,0].str.split('\t')
        tags['tags'] = [[i.split(':')[0].strip() for i in tag_list if i != ''] for tag_list in tags['tags']]
        #tags = tags['tags']

        clean_id = []
        new_tags = []
        for ind, tt in tags.iterrows():
            t = list(tt['tags'])
            tag = ''
            for i in range(len(t)-1):
                if 'hair' in t[i] or 'eye' in t[i]:
                    tag += t[i] + ' '
                    tag += t[-1]
                else:
                    continue;
            if tag == '': continue
            new_tags.append(tag)
            clean_id.append(ind)
        #print(clean_id, len(clean_id))
        onehot_matrix = skip_encode(new_tags)
        print(onehot_matrix)
        print("Encoded tag matrix shape:{}".format(onehot_matrix.shape))
        np.save('tag_clean_matrix', onehot_matrix)
    else:
        onehot_matrix = np.load('tag_clean_matrix.npy')
        print("Loading tag encoded matrix...shape:{}".format(onehot_matrix.shape))
    return onehot_matrix, clean_id

def load_one_hot_tags(path, preload=False):
    if not preload:
        tags = pd.read_csv(os.path.join(path, 'tags_clean.csv'), header=None, index_col=0)
        tags['tags'] = tags.iloc[:,0].str.split('\t')
        tags['tags'] = [[i.split(':')[0] for i in tag_list] for tag_list in tags['tags']]
        tags = tags['tags']
           
        # Train tokenizer
        tok = Tokenizer(num_words=1000)
        tags_lst = []
        for t in tags.values.tolist():
            tags_lst += t
        tok.fit_on_texts(tags_lst)
        with open("tok.pkl","wb") as l:
            pickle.dump(tok, l)
        
        clean_id = []
        new_tags = []
        for ind, tt in tags.iterrows():
            t = list(tt['tags'])
            tag = ''
            for i in range(len(t)-1):
                if 'hair' in t[i] or 'eye' in t[i]:
                    tag += t[i] + ' '
                    tag += t[-1]
                else:
                    continue;
            if tag == '': continue
            new_tags.append(tag)
            clean_id.append(ind)
        
        onehot_matrix = tok.texts_to_matrix(new_tags, mode='binary')
        print(onehot_matrix)
        print("Converting one hot encoding...One-hot matrix shape:{}".format(onehot_matrix.shape))
        np.save('tag_onehot_matrix', onehot_matrix)
    else:
        onehot_matrix = np.load('tag_onehot_matrix.npy')
        print("Loading one hot encoding matrix...shape:{}".format(onehot_matrix.shape))
    return onehot_matrix, clean_id

def load_tags(path, preload=False):
    if not preload:
        tags = pd.read_csv(os.path.join(path, 'tags_clean.csv'), header=None, index_col=0)
        tags['tags'] = tags.iloc[:,0].str.split('\t')
        tags['tags'] = [[i.split(':')[0].strip() for i in tag_list if i != ''] for tag_list in tags['tags']]
        tags = [i for i in tags['tags']]
        
        new_tags = []
        for t in tags:
            tag = ''
            for i in range(len(t)-1):
                tag += t[i] + ' '
            tag += t[-1]
            new_tags.append(tag)

        onehot_matrix = skip_encode(new_tags)
        print(onehot_matrix)
        print("Encoded tag matrix shape:{}".format(onehot_matrix.shape))
        np.save('tag_matrix', onehot_matrix)
    else:
        onehot_matrix = np.load('tag_matrix.npy')
        print("Loading tag encoded matrix...shape:{}".format(onehot_matrix.shape))
    return onehot_matrix

def skip_encode(tags):
    print("Start encode with skip-thought\n")
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(configuration.model_config(),
                        vocabulary_file=os.path.join('./skip_thoughts', 'pretrained', 'vocab.txt'),
                        embedding_matrix_file=os.path.join('./skip_thoughts', 'pretrained', 'embeddings.npy'),
                        checkpoint_path=os.path.join('./skip_thoughts', 'pretrained', 'model.ckpt-501424'))
    encoded = encoder.encode(tags, use_norm=False)
    print('Encoded by skip-thoughts model: shape {}\n'.format(encoded.shape)) 
    return encoded

if __name__ == "__main__":
    '''
    tags = pd.read_csv(os.path.join('./data', 'tags_clean.csv'), header=None, index_col=0)
    tags['tags'] = tags.iloc[:,0].str.split('\t')
    tags['tags'] = [[i.split(':')[0] for i in tag_list] for tag_list in tags['tags']]
    tags = tags['tags']
    
    new_tags = []
    for img_tag in tags.values.tolist():
        new_tag = ''
        for t in img_tag:
            new_tag = new_tag + t + ','
        new_tags.append(new_tag)
    skip_encode(new_tags)
    '''
    load_tags_clean('./data', preload=False)
    #load_data('./data', preload=False)
