import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from networks.Tiny_LaDeDa import tiny_ladeda
from networks.base_model import BaseModel, init_weights
from options.train_options import TrainOptions
from networks.tiny_trainer import TinyLaDeDaTrainer
import random

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print(f"seed: {seed}")


def get_info(dataset, type):
    X_patches, y_logits, y_img_labels = [], [], []
    label = 0 if type == "real" else 1
    for img_info in dataset:
        patches = img_info["patches"]  # [num of 9x9 patches in a given image, patch values]
        deepfake_scores = img_info["logits"]  # [num of 9x9 patches in a given image, deepfake score]
        X_patches.extend(patches)
        y_logits.extend(deepfake_scores.flatten())

        y_img_labels.append(label)

    X_patches, y_logits, y_img_labels = np.array(X_patches), np.array(y_logits), np.array(y_img_labels)
    return X_patches, y_logits, y_img_labels


def get_X_Y(dataset):
    X_real_patches, y_real_logits, y_real_labels = get_info(dataset["real"], "real")
    X_fake_patches, y_fake_logits, y_fake_labels = get_info(dataset["fake"], "fake")
    X = np.concatenate((X_real_patches, X_fake_patches), axis=0)
    Y = np.concatenate((y_real_logits, y_fake_logits), axis=0)
    labels = np.concatenate((y_real_labels, y_fake_labels), axis=0)
    return X, Y, labels


def load_data(path):
    loaded_data = np.load(path, allow_pickle=True)
    loaded_data = {key: loaded_data[key].tolist() for key in loaded_data.files}
    return loaded_data


def load_train_val(path):
    loaded_data = np.load(path, allow_pickle=True)
    patches = loaded_data["patches"]
    logits = loaded_data["logits"]
    return patches, logits

def train(X_train, y_train, X_val, y_val, val_img_labels):
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=729, shuffle=True)
    opt = TrainOptions().parse()
    model = TinyLaDeDaTrainer(opt, n_classes=1)
    net_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f'Model parameters {net_params:,d}')
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    val_loader = DataLoader(val_dataset, batch_size=729, shuffle=False)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    print(f"len train loader: {len(train_loader)}")
    print(f"len val loader: {len(val_loader)}")
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for i, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()


        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
        # Validation
        model.eval()
        val_logits = []
        with torch.no_grad():
            for patches, logits in val_loader:
                val_outputs = model.model(patches.cuda()).squeeze().detach().cpu().numpy()
                # getting img deepfake score
                img_logit = val_outputs.mean()
                val_logits.extend(torch.sigmoid(torch.tensor(img_logit)).flatten().tolist())

        val_logits = np.array(val_logits)
        acc = accuracy_score(val_img_labels, val_logits)
        wandb.log({'accuracy': acc, 'ap': ap}, commit=False)
        wandb.log({'epoch': epoch})
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()



if __name__ == '__main__':
    seed_torch(seed=42)

    # the distillation training and validation sets
    train_path = "patches_logits_train_set.npz"
    val_path = "patches_logits_val_set.npz"

    print("loading train")
    train_data = load_data(train_path)
    X_train, y_train, train_img_labels = get_X_Y(train_data)

    print("loading validation")
    val_data = load_data(val_path)
    X_val, y_val, val_img_labels = get_X_Y(val_data)

    train(X_train, y_train, X_val, y_val, val_img_labels)


