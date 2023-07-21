"""adapted from https://github.com/singlasahil14/salient_imagenet"""

import cv2
import numpy as np
import torch


def generate_penultimate_features(model, data_loader, device=torch.device("cuda")):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    features_all = []
    total = 0
    targets_all = []
    for images, target, _ in data_loader:
        images = images.to(device)
        model.flatten.register_forward_hook(get_activation("flatten"))
        output = model(images)
        features = activation["flatten"]
        features = features.detach()
        total += len(images)

        features_all.append(features.detach().cpu().numpy())
        targets_all.append(target.detach().cpu().numpy())

    features_all = np.concatenate(features_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    return features_all, targets_all


def compute_layer_feature_maps(
    images, model, layer_name="layer4", device=torch.device("cuda")
):
    images = images.to(device)

    x = images
    for name, module in model._modules.items():
        x = model.relu(module(x))
        if name == layer_name:
            break
        x = model.pool(x)
    return x


def compute_heatmaps(imgs, masks):
    heatmaps = []
    for img, mask in zip(imgs, masks):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap + np.float32(img)
        heatmap = heatmap / np.max(heatmap)
        heatmaps.append(heatmap)
    heatmaps = np.stack(heatmaps, axis=0)
    heatmaps = torch.from_numpy(heatmaps).permute(0, 3, 1, 2)
    return heatmaps


def compute_nams(
    model, images, feature_index, layer_name="layer4", device=torch.device("cuda")
):
    b_size = images.shape[0]
    # feature_maps = compute_feature_maps(images, model, layer_name=layer_name)
    feature_maps = compute_layer_feature_maps(
        images, model, layer_name=layer_name, device=device
    )
    # nams = (feature_maps[:, feature_index, ]).detach()
    nams = (feature_maps[:, feature_index, :, :]).detach()
    nams_flat = nams.view(b_size, -1)
    nams_max, _ = torch.max(nams_flat, dim=1, keepdim=True)
    nams_flat = nams_flat / nams_max
    nams = nams_flat.view_as(nams)

    nams_resized = []
    for nam in nams:
        nam = nam.cpu().numpy()
        nam = cv2.resize(nam, images.shape[2:])
        nams_resized.append(nam)
    nams = np.stack(nams_resized, axis=0)
    nams = torch.from_numpy(1 - nams)
    return nams


def load_images(indices, dataset):
    img_list = []
    for idx in indices:
        img = dataset.__getitem__(idx)[0]
        img_list.append(img)
    img_tensor = torch.stack(img_list, dim=0)
    return img_tensor


def create_images(indices_high, feature_index, dataset, robust_model, device):
    images_highest = load_images(indices_high, dataset)
    images_nams = compute_nams(
        robust_model, images_highest, feature_index, layer_name="conv5", device=device
    )

    images_heatmaps = compute_heatmaps(images_highest.permute(0, 2, 3, 1), images_nams)
    return images_highest, images_heatmaps


def get_topk_predictive_features(class_index, robust_model, robust_features, k=5):
    W = (robust_model.fc.weight).detach().cpu().numpy()
    W_class = W[class_index : class_index + 1, :]
    FI_values = np.mean(robust_features * W_class, axis=0)

    features_indices = np.argsort(-FI_values)[:k]
    return features_indices
