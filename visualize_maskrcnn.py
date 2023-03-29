import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional
import time

def select_top_predictions(predictions, threshold):
    idx = (predictions["scores"] > threshold).nonzero().squeeze(1)
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    """
    Simple function that adds fixed colors depending on the class
    """
    if palette is None:
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions["labels"]
    boxes = predictions['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions["masks"].ge(0.5).mul(255).byte().numpy()
    labels = predictions["labels"]

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 2)

    composite = image

    return composite



def overlay_class_names(image, predictions, categories, target, action):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions["scores"].tolist()
    labels = predictions["labels"].tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions['boxes']

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        box=box.tolist()
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1
        )
    
    # put target + action text    
    if target:
        cv2.putText(image, target + action, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
    return image
    
    
def predict(img, output, categories, target=False, action =False):
    cv_img = np.array(img)[:, :, [2, 1, 0]]
    top_predictions = select_top_predictions(output, 0.7)
    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
    result = cv_img.copy()
    # result = overlay_boxes(result, top_predictions)
    if 'masks' in top_predictions:
        result = overlay_mask(result, top_predictions)
    result = overlay_class_names(result, top_predictions, categories, target, action)
    return result, output, top_predictions
