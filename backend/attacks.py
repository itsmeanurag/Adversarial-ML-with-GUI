import numpy as np
import tensorflow as tf

try:
    from art.estimators.classification import TensorFlowV2Classifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, BoundaryAttack, HopSkipJump
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

def fgsm_attack(model, img, epsilon=0.1, dataset=None):
    if ART_AVAILABLE and dataset is not None:
        ds = dataset.lower()
        nb_classes = {"mnist":10, "cifar100":100, "gtsrb":43}.get(ds, 1000)
        input_shape = img.shape
        preds = model.predict(img[np.newaxis, ...])
        pred_class = np.argmax(preds)
        label_onehot = np.zeros((1, nb_classes), dtype=np.float32)
        label_onehot[0, pred_class] = 1.0
        art_classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            channels_first=False,
        )
        adv_img = FastGradientMethod(art_classifier, eps=epsilon).generate(img[np.newaxis, ...], y=label_onehot)
        if ds == "imagenet":
            return np.clip(adv_img[0], -128.0, 128.0)
        else:
            return np.clip(adv_img[0], 0, 1)

    # Manual FGSM (fallback, always for ImageNet or if ART unavailable)
    image = tf.convert_to_tensor(img[None], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        label = tf.argmax(prediction, axis=1)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    adv_image = image + epsilon * signed_grad

    if dataset is not None and dataset.lower() == "imagenet":
        adv_image = tf.clip_by_value(adv_image, -128.0, 128.0)
    else:
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)

    return adv_image.numpy()[0]

def pgd_attack(model, img, epsilon=0.1, alpha=0.01, iters=40, dataset=None):
    if ART_AVAILABLE and dataset is not None:
        ds = dataset.lower()
        nb_classes = {"mnist":10, "cifar100":100, "gtsrb":43}.get(ds, 1000)
        input_shape = img.shape
        preds = model.predict(img[np.newaxis, ...])
        pred_class = np.argmax(preds)
        label_onehot = np.zeros((1, nb_classes), dtype=np.float32)
        label_onehot[0, pred_class] = 1.0
        art_classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            channels_first=False,
        )
        adv_img = ProjectedGradientDescent(
            art_classifier, eps=epsilon, eps_step=alpha, max_iter=iters
        ).generate(img[np.newaxis, ...], y=label_onehot)
        if ds == "imagenet":
            return np.clip(adv_img[0], -128.0, 128.0)
        else:
            return np.clip(adv_img[0], 0, 1)
    return img

def boundary_attack(model, img, dataset=None):
    if ART_AVAILABLE and dataset is not None:
        ds = dataset.lower()
        nb_classes = {"mnist": 10, "cifar100": 100, "gtsrb": 43}.get(ds, 1000)
        art_classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=nb_classes,
            input_shape=img.shape,
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            channels_first=False,
        )
        attack = BoundaryAttack(art_classifier, targeted=False)
        adv_img = attack.generate(img[np.newaxis, ...])
        if ds == "imagenet":
            return np.clip(adv_img[0], -128.0, 128.0)
        else:
            return np.clip(adv_img[0], 0, 1)
    return img

def hopskipjump_attack(model, img, dataset=None):
    if ART_AVAILABLE and dataset is not None:
        ds = dataset.lower()
        nb_classes = {"mnist": 10, "cifar100": 100, "gtsrb": 43}.get(ds, 1000)
        art_classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=nb_classes,
            input_shape=img.shape,
            loss_object=tf.keras.losses.CategoricalCrossentropy(),
            channels_first=False,
        )
        attack = HopSkipJump(art_classifier)
        adv_img = attack.generate(img[np.newaxis, ...])
        if ds == "imagenet":
            return np.clip(adv_img[0], -128.0, 128.0)
        else:
            return np.clip(adv_img[0], 0, 1)
    return img