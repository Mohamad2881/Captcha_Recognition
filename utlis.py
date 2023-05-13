import torch

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from ctc_decoder import best_path, beam_search

import dataset


DATA_DIR = './Data/images'
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 75
NUM_WORKERS = 8
DEVICE = "cuda"

def get_datasets():
    images_path = glob.glob(os.path.join(DATA_DIR, '*.png'))
    labels_orig = [path.split('/')[-1][:-4] for path in images_path]

    # convert list of strings to list of lists of chars
    # ['ads', 'rew', 'jto'] --> [['a', 'd', 's'], ['r', 'e', 'w'], ['j', 't', 'o']]
    labels = [[*label] for label in labels_orig]

    # labels = [c for label in labels_orig for c in label]
    le = preprocessing.LabelEncoder()
    le.fit(np.array(labels).flatten())

    labels_encoded = [le.transform(lbl) for lbl in labels]

    # we add one to keep zero for 'Unkown'
    labels_encoded = np.array(labels_encoded)

    train_paths, test_paths, train_labels, test_labels, train_labels_orig, test_labels_orig =\
        train_test_split(images_path, labels_encoded, labels_orig, test_size=0.2, random_state=42)

    train_dataset = dataset.Dataset(images_path=train_paths, labels=train_labels,
                                            resize=(IMAGE_HEIGHT, IMAGE_WIDTH))


    test_dataset = dataset.Dataset(images_path=test_paths, labels=test_labels,
                                            resize=(IMAGE_HEIGHT, IMAGE_WIDTH))

    # print(len(le.classes_))
    # print(labels_encoded)

    return train_dataset, test_dataset, test_labels_orig, le


from tqdm import tqdm


def train_one_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0

    # for every batch
    for data in tqdm(data_loader, total=len(data_loader)):
        # add data to device
        for k, v in data.items():
            data[k] = v.to(DEVICE)

        model.zero_grad()

        _, loss = model(**data)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_one_epoch(model, data_loader):
    model.eval()
    total_loss = 0
    all_pred = []

    with torch.no_grad():
        # for every batch
        for data in tqdm(data_loader, total=len(data_loader)):
            # add data to device
            for k, v in data.items():
                data[k] = v.to(DEVICE)

            preds, loss = model(**data)
            total_loss += loss.item()
            all_pred.append(preds)

        return total_loss / len(data_loader), all_pred


def decode_preds(preds, lbl_enc):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()

    all_preds = []
    # for each in image in the batch
    for j in range(preds.shape[0]):
        temp = ''
        # for every output element in output vector
        # print(preds[j, :])
        for k in preds[j, :]:

            # we append 1 before
            # k = k - 1
            if k == 19:
                temp += '*'
            else:
                # transform back the index to a char
                temp += lbl_enc.inverse_transform([k])[0]
        all_preds.append(temp)

    return all_preds




def beam_decoder(rnn_outputs, chars):
    # rnn_outputs-->  TxBxC

    outputs = []
    rnn_outputs = rnn_outputs.permute(1, 0, 2)
    rnn_outputs = torch.softmax(rnn_outputs, 2)
    rnn_outputs = rnn_outputs.detach().cpu().numpy()

    for mat in rnn_outputs:
        outputs.append(beam_search(mat, chars))

    return outputs


def predict(model, test_dataloader, chars):
    for batch in test_dataloader:

        for k, v in batch.items():
            batch[k] = v.to(DEVICE)

        images, labels = batch.values()

        with torch.no_grad():
            preds, _ = model(images, labels)

        decoded_preds = beam_decoder(preds, chars)

        imgs = np.transpose(images.cpu().numpy(), (0, 2, 3, 1))
        imgs = (imgs).astype(np.uint8)

        plot_predictions(imgs, decoded_preds, nrows=None, ncols=4)
        plt.show()



def plot_predictions(imgs, labels, nrows=None, ncols=5):
    if nrows is None:
        nrows = int(np.ceil(len(labels) // ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 2))
    for (img, lbl, ax) in zip(imgs, labels, axs.flatten()):
        ax.imshow(np.invert(img))
        ax.set_title(f'Predicted Label: {lbl}')