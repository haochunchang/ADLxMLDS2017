## Preprocess functions
import os, pickle, glob
import pandas as pd
import numpy as np
from skimage import io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(path, preload=False):
    if not preload:
        img_path = os.path.join(path, 'faces')
        img_all_path = sorted(glob.glob(os.path.join(img_path, '*.jpg')), 
                                key=lambda x: int(x.split('/')[-1].split('.')[0]))
        imgs = io.imread_collection(img_all_path)
        imgs = io.concatenate_images(imgs)
        print("Image matrix shape:{}".format(imgs.shape))
        np.save('img_matrix', imgs)
    else:
        imgs = np.load('img_matrix.npy')
        print("Loading true img matrix...shape:{}".format(imgs.shape))
    return imgs

def load_tags(path, preload=False):
    if not preload:
        tags = pd.read_csv(os.path.join(path, 'tags_clean.csv'), header=None, index_col=0)
        tags['tags'] = tags.iloc[:,0].str.split('\t')
        tags['tags'] = [[i.split(':')[0] for i in tag_list] for tag_list in tags['tags']]
        tags = tags['tags']
           
        # Train tokenizer
        tok = Tokenizer(num_words=5000)
        tags_lst = []
        for t in tags.values.tolist():
            tags_lst += t
        tok.fit_on_texts(tags_lst)
        with open("tok.pkl","wb") as l:
            pickle.dump(tok, l)
        
        # Convert tags to word index
        new_tags = []
        for img_tag in tags.values.tolist():
            new_tag = ''
            for t in img_tag:
                new_tag = new_tag + t + ','
            new_tags.append(new_tag)
        
        onehot_matrix = tok.texts_to_matrix(new_tags, mode='binary')
        print(onehot_matrix)
        print("Converting one hot encoding...One-hot matrix shape:{}".format(onehot_matrix.shape))
        np.save('tag_onehot_matrix', onehot_matrix)
    else:
        onehot_matrix = np.load('tag_onehot_matrix.npy')
        print("Loading one hot encoding matrix...shape:{}".format(onehot_matrix.shape))
    return onehot_matrix


if __name__ == "__main__":
    load_tags('./data', preload=False)
    #load_data('./data', preload=True)
