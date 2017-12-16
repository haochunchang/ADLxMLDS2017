## Preprocess functions
import os, pickle, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from skimage import io

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
    
        # convert to onehot encoding
        le = load_label_encoder(tags)
        int_tags = tags.apply(le.transform)
        onehot_tags = int_tags.apply(lambda x: convert_onehot(x, len(le.classes_)))
        onehot_matrix = np.array(onehot_tags.values.tolist())
        print("Converting one hot encoding...One-hot matrix shape:{}".format(onehot_matrix.shape))
        np.save('tag_onehot_matrix', onehot_matrix)
    else:
        onehot_matrix = np.load('tag_onehot_matrix.npy')
        print("Loading one hot encoding matrix...shape:{}".format(onehot_matrix.shape))
    return onehot_matrix

def load_label_encoder(label):
    try:
        with open("le.pkl",'rb') as l:
            return pickle.load(l);
    except:
        le = LabelEncoder()
        labels = []
        for dlabel in label.values.tolist():
            labels += dlabel
        le.fit(labels)
        with open("le.pkl","wb") as l:
            pickle.dump(le, l)
        return le

def convert_onehot(label, num_class):
    # input: example: [3, 14], integer labels
    newlabel = np.zeros(num_class)
    for l in label:
        newlabel[l] = 1
    return newlabel
 
if __name__ == "__main__":
    load_tags('./data', preload=True)
