import torch
import fcn_model
import fcn_dataset
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F


# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
num_classes = 32
model = fcn_model.FCN8s(num_classes).to(device)

# Define the dataset and dataloader
images_dir_train = "train/"
labels_dir_train = "train_labels/"
class_dict_path = "class_dict.csv"
resolution = (384, 512)
batch_size = 16
num_epochs = 50

camvid_dataset_train = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_train, labels_dir=labels_dir_train, class_dict_path=class_dict_path, resolution=resolution, crop=True)
dataloader_train = torch.utils.data.DataLoader(camvid_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

images_dir_val = "val/"
labels_dir_val = "val_labels/"
camvid_dataset_val = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_val, labels_dir=labels_dir_val, class_dict_path=class_dict_path, resolution=resolution, crop=False)
dataloader_val = torch.utils.data.DataLoader(camvid_dataset_val, batch_size=1, shuffle=False, num_workers=4, drop_last=False)


images_dir_test = "test/"
labels_dir_test = "test_labels/"
camvid_dataset_test = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_test, labels_dir=labels_dir_test, class_dict_path=class_dict_path, resolution=resolution, crop=False)
dataloader_test = torch.utils.data.DataLoader(camvid_dataset_test, batch_size=1, shuffle=False, num_workers=4, drop_last=False)


def loss_fn(outputs, labels):

    loss_fc=nn.CrossEntropyLoss()

    return loss_fc(outputs, labels)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


#THIS (eval_model1) IS MY ORIGINAL IDEA TO IMPELEMENT eval_model. BUT THIS IS TOO INEFFICIENT. SO I MADE IT MORE EFFICIENT IN eval_model

def eval_model1(model, dataloader, device, save_pred=True):
    model.eval()
    loss_list = []

    confusion_matrix=dict()

    for i in range(num_classes):
        confusion_matrix[i]=dict()
        for j in range(num_classes):
            confusion_matrix[i][j]=0

    if save_pred:
        pred_list = []

    #total_correct_pixels = 0
    #total_pixels = 0
    with torch.no_grad():

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            if save_pred:
                pred_list.append(predicted.cpu().numpy())

            #total_pixels += labels.numel()
            #total_correct_pixels += torch.sum(predicted == labels).item()

            for i in range(len(predicted)):
                for j in range(len(predicted[0])):
                  for k in range(len(predicted[0][0])):
                        confusion_matrix[predicted[i][j][k].item()][labels[i][j][k].item()]+=1


        t=dict()
        for i in range(num_classes):
              t[i]=0
              for j in range(num_classes):
                  t[i]+=confusion_matrix[i][j]


        pa_num=sum([confusion_matrix[ind][ind] for ind in range(num_classes)])
        pa_den=sum(t.values())
        pixel_acc = pa_num/pa_den


        miou=sum([confusion_matrix[ind][ind]/(t[ind] + sum(confusion_matrix[ind0][ind] - confusion_matrix[ind][ind] for ind0 in range(num_classes))) for ind in range(num_classes)])
        mean_iou = miou*(1/num_classes)

        fiou=sum([(confusion_matrix[ind][ind]*t[ind])/(t[ind] + sum(confusion_matrix[ind0][ind] - confusion_matrix[ind][ind] for ind0 in range(num_classes))) for ind in range(num_classes)])
        freq_iou = (1/pa_den)*fiou

        loss = sum(loss_list) / len(loss_list)
        print('Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f}, Loss: {:.4f}'.format(pixel_acc, mean_iou, freq_iou, loss))

    if save_pred:
        pred_list = np.concatenate(pred_list, axis=0)
        np.save('test_pred.npy', pred_list)
    model.train()




def eval_model(model, dataloader, device, save_pred=  False):
    model.eval()
    loss_list = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    if save_pred:
        pred_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_list.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            if save_pred:
                pred_list.append(predicted.cpu().numpy())

            # Update confusion matrix
            predicted_flat = predicted.cpu().view(-1)  # Move to CPU
            labels_flat = labels.cpu().view(-1)  # Move to CPU
            confusion_matrix += np.bincount(num_classes * labels_flat.numpy() + predicted_flat.numpy(), minlength=num_classes**2).reshape(num_classes, num_classes)

        # Compute metrics from confusion matrix
        correct_pixels = np.diag(confusion_matrix).sum()
        total_pixels = confusion_matrix.sum()

        pixel_acc = correct_pixels / total_pixels

        class_totals = confusion_matrix.sum(axis=1)
        class_intersection = np.diag(confusion_matrix)
        class_union = class_totals + confusion_matrix.sum(axis=0) - class_intersection

        class_iou = np.nan_to_num(class_intersection / class_union)
        mean_iou = np.mean(class_iou)

        class_freq_iou = class_iou * class_totals
        freq_iou = class_freq_iou.sum() / total_pixels

        loss = sum(loss_list) / len(loss_list)
        print('Pixel accuracy: {:.4f}, Mean IoU: {:.4f}, Frequency weighted IoU: {:.4f}, Loss: {:.4f}'.format(pixel_acc, mean_iou, freq_iou, loss))

    if save_pred:
        pred_list = np.concatenate(pred_list, axis=0)
        np.save('test_pred.npy', pred_list)
    model.train()



def visualize_model(model, dataloader, device):
    log_dir = "vis/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    cls_dict = dataloader.dataset.class_dict.copy()
    cls_list = [cls_dict[i] for i in range(len(cls_dict))]
    model.eval()
    with torch.no_grad():
        for ind, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            images_vis = fcn_dataset.rev_normalize(images)
            # Save the images and labels
            img = images_vis[0].permute(1, 2, 0).cpu().numpy()
            img = img * 255
            img = img.astype('uint8')
            label = labels[0].cpu().numpy()
            pred = predicted[0].cpu().numpy()

            label_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            pred_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            for j in range(len(cls_list)):
                mask = label == j
                label_img[mask] = cls_list[j][0]
                mask = pred == j
                pred_img[mask] = cls_list[j][0]
            # horizontally concatenate the image, label, and prediction, and save the visualization
            vis_img = np.concatenate([img, label_img, pred_img], axis=1)
            vis_img = Image.fromarray(vis_img)
            vis_img.save(os.path.join(log_dir, 'img_{:04d}.png'.format(ind)))

    model.train()




# Train the model
loss_list = []
for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(dataloader_train):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        #print(f"the size of outputs is {outputs.size()}")
        #print(f"the size of labels is {labels.size()}")

        loss = loss_fn(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader_train), sum(loss_list)/len(loss_list)))
            loss_list = []

    # eval the model
    eval_model(model, dataloader_val, device)

print('='*20)
print('Finished Training, evaluating the model on the test set')
eval_model(model, dataloader_test, device, save_pred=True)

print('='*20)
print('Visualizing the model on the test set, the results will be saved in the vis/ directory')
visualize_model(model, dataloader_test, device)









