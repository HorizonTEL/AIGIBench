import torch
import numpy as np
from networks.LaDeDa import LaDeDa9
from networks.Tiny_LaDeDa import tiny_ladeda
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_auc_score, precision_score, recall_score
from data import create_dataloader
import torch.nn as nn

def validate(model, opt):
    import time
    data_loader, _ = create_dataloader(opt)
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            img_input = img.cuda()
            y_pred.extend(model(img_input).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred > 0.5)
    recall = recall_score(y_true, y_pred > 0.5)
    return acc, ap, r_acc, f_acc


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)
    model = LaDeDa(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, r_acc, f_acc, auc, precision, recall = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)


