import numpy as np
from tensorflow.keras.applications.resnet50 import decode_predictions

def predict_image(model, img, dataset=None):
    preds = model.predict(img[np.newaxis, ...])
    if dataset == "imagenet":
        # Return the full softmax array for decode_predictions, plus the confidence
        return preds, float(np.max(preds))
    else:
        pred_class = int(np.argmax(preds))
        conf = float(np.max(preds))
        return pred_class, conf