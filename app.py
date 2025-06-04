import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_RUN_ON_SAVE"] = "false"

import streamlit as st
import numpy as np
from PIL import Image
import random
import requests
from io import BytesIO

from backend.models import (
    load_mnist_model,
    load_cifar100_model,
    load_imagenet_model,
    load_gtsrb_model,
)
from backend.dataset_utils import (
    get_random_mnist_sample,
    get_random_cifar100_sample,
    get_random_gtsrb_sample,
)
from backend.predict import predict_image
from backend.attacks import fgsm_attack, pgd_attack, boundary_attack, hopskipjump_attack
from backend.class_names import MNIST_CLASSES, CIFAR100_CLASSES, GTSRB_CLASSES

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

IMAGENET_SAMPLE_URLS = [
    "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg",
]

def get_random_imagenet_sample_online():
    valid = False
    attempts = 0
    max_attempts = 10
    headers = {"User-Agent": "adversarial-ml-art-gui/1.0"}
    while not valid and attempts < max_attempts:
        url = random.choice(IMAGENET_SAMPLE_URLS)
        try:
            response = requests.get(url, timeout=5, headers=headers)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB").resize((224, 224))
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            img_array = preprocess_input(img_array.astype("float32"))
            valid = True
            return img_array, url
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            attempts += 1
    raise RuntimeError("Could not fetch a valid ImageNet sample image after multiple tries.")

st.set_page_config(page_title="Adversarial ML Demo", layout="wide")

dataset = st.selectbox(
    "Choose Dataset",
    ["MNIST", "CIFAR-100", "ImageNet", "GTSRB"],
    index=0,
    help="Choose which dataset to demo",
)
is_mnist = dataset == "MNIST"
is_cifar = dataset == "CIFAR-100"
is_imagenet = dataset == "ImageNet"
is_gtsrb = dataset == "GTSRB"

with st.sidebar:
    st.title("🧪 Adversarial ML Demo")
    if is_mnist:
        st.markdown(
            """
            - **Dataset:** MNIST (10 handwritten digits)
            - **Features:**  
                - Random digit prediction  
                - Generate and visualize adversarial attacks  
                - Easily fetch new digit examples
            """
        )
        st.markdown("> **About MNIST:**\n> Classic dataset of 28x28 grayscale images of handwritten digits (0-9).")
    elif is_cifar:
        st.markdown(
            """
            - **Dataset:** CIFAR-100 (100 object categories)
            - **Features:**  
                - Random object prediction  
                - Generate and visualize adversarial attacks  
                - Easily fetch new image examples
            """
        )
        st.markdown("> **About CIFAR-100:**\n> 32x32 color images from 100 diverse object categories (animals, vehicles, household objects, etc).")
        st.caption("Tip: CIFAR-100 is much harder than MNIST and model accuracy will be lower.")
    elif is_gtsrb:
        st.markdown(
            """
            - **Dataset:** GTSRB (German Traffic Sign)
            - **Features:**  
                - Random traffic sign prediction  
                - Generate and visualize adversarial attacks  
                - Easily fetch new image examples  
            """
        )
        st.markdown("> **About GTSRB:**\n> 48x48 color images of 43 different traffic signs.")
    else:
        st.markdown(
            """
            - **Dataset:** ImageNet (1000 object categories)
            - **Features:**  
                - Predict on ImageNet sample or uploaded image  
                - Generate and visualize adversarial attacks  
                - Easily fetch new sample images
            """
        )

if is_mnist:
    CLASS_NAMES = MNIST_CLASSES
    get_sample = get_random_mnist_sample
    IMG_SHAPE = (28, 28, 1)
    FGSM_DATASET_KEY = "mnist"
    FGSM_EPSILON = 0.25
    PGD_EPSILON = 0.25
    PGD_ALPHA = 0.01
    PGD_ITERS = 40
elif is_cifar:
    CLASS_NAMES = CIFAR100_CLASSES
    get_sample = get_random_cifar100_sample
    IMG_SHAPE = (32, 32, 3)
    FGSM_DATASET_KEY = "cifar100"
    FGSM_EPSILON = 0.05
    PGD_EPSILON = 0.05
    PGD_ALPHA = 0.01
    PGD_ITERS = 40
elif is_gtsrb:
    CLASS_NAMES = GTSRB_CLASSES
    get_sample = get_random_gtsrb_sample
    IMG_SHAPE = (48, 48, 3)  # Updated for 48x48
    FGSM_DATASET_KEY = "gtsrb"
    FGSM_EPSILON = 0.05
    PGD_EPSILON = 0.05
    PGD_ALPHA = 0.01
    PGD_ITERS = 40
else:
    CLASS_NAMES = None
    get_sample = get_random_imagenet_sample_online
    IMG_SHAPE = (224, 224, 3)
    FGSM_DATASET_KEY = "imagenet"
    FGSM_EPSILON = 0.25
    PGD_EPSILON = 0.25
    PGD_ALPHA = 0.01
    PGD_ITERS = 40

def get_model(dataset):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    if dataset == "MNIST":
        return load_mnist_model()
    elif dataset == "CIFAR-100":
        return load_cifar100_model()
    elif dataset == "GTSRB":
        return load_gtsrb_model()
    else:
        return load_imagenet_model()  # Should still use tf.keras.applications

model = get_model(dataset)

if "dataset" not in st.session_state or st.session_state.dataset != dataset:
    if is_imagenet:
        img, url = get_sample()
        st.session_state.img = img
        st.session_state.imagenet_url = url
    else:
        img, label = get_sample()
        # Ensure image is the correct size for GTSRB
        if is_gtsrb and (img.shape[0] != 48 or img.shape[1] != 48):
            img = Image.fromarray((img * 255).astype("uint8"))
            img = img.resize((48, 48))
            img = np.array(img).astype("float32") / 255.0
        st.session_state.img = img
        st.session_state.label = label
    st.session_state.pred = None
    st.session_state.conf = None
    st.session_state.adv_img = None
    st.session_state.adv_pred = None
    st.session_state.adv_conf = None
    st.session_state.dataset = dataset
    if "user_uploaded" in st.session_state:
        del st.session_state["user_uploaded"]

col_img, col_controls = st.columns([2, 1], gap="large")

if is_imagenet:
    with col_img:
        mean = np.array([103.939, 116.779, 123.68])
        img_disp = st.session_state.img + mean
        img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)
        st.image(img_disp, width=250, caption=(
            f"ImageNet sample: [Source]({st.session_state.imagenet_url})"
            if "user_uploaded" not in st.session_state
            else "User uploaded image"
        ))
        if st.session_state.get("adv_img") is not None:
            adv_disp = st.session_state.adv_img + mean
            adv_disp = np.clip(adv_disp, 0, 255).astype(np.uint8)
            st.image(adv_disp, width=250, caption="Adversarial Example")

    with col_controls:
        st.markdown("**Choose an image:**")
        if st.button("🔀 Random Image"):
            img_array, url = get_random_imagenet_sample_online()
            st.session_state.img = img_array
            st.session_state.imagenet_url = url
            st.session_state.pred = None
            st.session_state.adv_pred = None
            st.session_state.adv_img = None
            if "user_uploaded" in st.session_state:
                del st.session_state["user_uploaded"]

        uploaded = st.file_uploader("Or upload an image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded).resize((224, 224))
            img_array = np.array(img)
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
            img_array = preprocess_input(img_array.astype("float32"))
            st.session_state.img = img_array
            st.session_state.pred = None
            st.session_state.adv_pred = None
            st.session_state.adv_img = None
            st.session_state.imagenet_url = "User upload"
            st.session_state["user_uploaded"] = True

        if st.button("🔍 Predict"):
            preds, conf = predict_image(model, st.session_state.img, dataset=FGSM_DATASET_KEY)
            st.session_state.pred = preds
            st.session_state.conf = conf

        attack_type = st.radio(
            "Choose Attack Type",
            ["FGSM", "PGD", "Boundary", "HopSkipJump"],
            horizontal=True
        )
        if st.button("🚨 Generate Adversarial Example"):
            if st.session_state.get("pred") is None:
                preds, conf = predict_image(model, st.session_state.img, dataset=FGSM_DATASET_KEY)
                st.session_state.pred = preds
                st.session_state.conf = conf
            adv_img = None
            if attack_type == "FGSM":
                adv_img = fgsm_attack(
                    model,
                    st.session_state.img,
                    epsilon=FGSM_EPSILON,
                    dataset=FGSM_DATASET_KEY,
                )
            elif attack_type == "PGD":
                adv_img = pgd_attack(
                    model,
                    st.session_state.img,
                    epsilon=PGD_EPSILON,
                    alpha=PGD_ALPHA,
                    iters=PGD_ITERS,
                    dataset=FGSM_DATASET_KEY,
                )
            elif attack_type == "Boundary":
                try:
                    adv_img = boundary_attack(
                        model,
                        st.session_state.img,
                        dataset=FGSM_DATASET_KEY,
                    )
                except Exception as e:
                    st.error(f"Boundary Attack failed: {e}")
                    adv_img = None
            elif attack_type == "HopSkipJump":
                try:
                    adv_img = hopskipjump_attack(
                        model,
                        st.session_state.img,
                        dataset=FGSM_DATASET_KEY,
                    )
                except Exception as e:
                    st.error(f"HopSkipJump Attack failed: {e}")
                    adv_img = None
            if adv_img is not None:
                adv_preds, adv_conf = predict_image(model, adv_img, dataset=FGSM_DATASET_KEY)
                st.session_state.adv_img = adv_img
                st.session_state.adv_pred = adv_preds
                st.session_state.adv_conf = adv_conf

    col_pred, col_adv = st.columns(2, gap="large")
    with col_pred:
        st.subheader("Prediction Results")
        if st.session_state.get("pred") is not None:
            decoded = decode_predictions(st.session_state.pred, top=3)[0]
            for i, (cls_id, name, conf) in enumerate(decoded):
                st.success(f"Top {i+1}: {name} ({conf:.2f})")
        else:
            st.info("Click **Predict** to see the model's prediction.")
    with col_adv:
        st.subheader("Adversarial Results")
        if st.session_state.get("adv_img") is not None:
            decoded = decode_predictions(st.session_state.adv_pred, top=3)[0]
            for i, (cls_id, name, conf) in enumerate(decoded):
                st.warning(f"Top {i+1}: {name} ({conf:.2f})")
            orig_decoded = decode_predictions(st.session_state.pred, top=1)[0][0][1]
            adv_decoded = decoded[0][1]
            if orig_decoded != adv_decoded:
                st.error(f"Prediction changed: {orig_decoded} → {adv_decoded}")
        else:
            st.info("Click **Generate Adversarial Example** to see the attack effect.")
else:
    with col_img:
        st.subheader(f"🖼️ Current {dataset} Image")
        img_disp = st.session_state.img
        if is_mnist or is_gtsrb:
            if img_disp.shape[-1] == 1:
                img_disp = np.repeat(img_disp, 3, axis=-1)
            img_disp = (img_disp * 255).astype(np.uint8)
        elif is_cifar:
            img_disp = (img_disp * 255).astype(np.uint8)
        st.image(img_disp, width=250, caption=(
            f"True Label: {CLASS_NAMES[st.session_state.label]}"
            if "user_uploaded" not in st.session_state
            else "User uploaded image"
        ))
        if st.session_state.get("adv_img") is not None:
            st.subheader("⚡ Adversarial Image")
            adv_disp = st.session_state.adv_img
            if is_mnist or is_gtsrb:
                if adv_disp.shape[-1] == 1:
                    adv_disp = np.repeat(adv_disp, 3, axis=-1)
                adv_disp = (adv_disp * 255).astype(np.uint8)
            elif is_cifar:
                adv_disp = (adv_disp * 255).astype(np.uint8)
            st.image(adv_disp, width=250, caption="Adversarial Example (after attack)")

    with col_controls:
        st.subheader("🔎 Actions")
        st.write("Choose what to do with the current image:")
        uploaded = st.file_uploader(
            f"Or upload a {'28x28 grayscale' if is_mnist else ('48x48 color' if is_gtsrb else '32x32 color')} image (PNG/JPEG)",
            type=["jpg", "jpeg", "png"],
            key=f"upload_{dataset.lower()}"
        )
        if uploaded is not None:
            img = Image.open(uploaded)
            if is_mnist:
                img = img.convert("L").resize((28, 28))
                img_array = np.array(img).astype("float32") / 255.0
                img_array = img_array[..., np.newaxis]
            elif is_cifar:
                img = img.convert("RGB").resize((32, 32))
                img_array = np.array(img).astype("float32") / 255.0
            elif is_gtsrb:
                img = img.convert("RGB").resize((48, 48))
                img_array = np.array(img).astype("float32") / 255.0
            st.session_state.img = img_array
            st.session_state.pred = None
            st.session_state.conf = None
            st.session_state.adv_img = None
            st.session_state.adv_pred = None
            st.session_state.adv_conf = None
            st.session_state["user_uploaded"] = True

        if st.button("🔍 Predict"):
            pred_class, conf = predict_image(model, st.session_state.img, dataset=FGSM_DATASET_KEY)
            st.session_state.pred = pred_class
            st.session_state.conf = conf

        attack_type = st.radio(
            "Choose Attack Type",
            ["FGSM", "PGD", "Boundary", "HopSkipJump"],
            horizontal=True
        )
        if st.button("🚨 Generate Adversarial Example"):
            if st.session_state.get("pred") is None:
                pred_class, conf = predict_image(model, st.session_state.img, dataset=FGSM_DATASET_KEY)
                st.session_state.pred = pred_class
                st.session_state.conf = conf
            adv_img = None
            if attack_type == "FGSM":
                adv_img = fgsm_attack(
                    model,
                    st.session_state.img,
                    epsilon=FGSM_EPSILON,
                    dataset=FGSM_DATASET_KEY,
                )
            elif attack_type == "PGD":
                adv_img = pgd_attack(
                    model,
                    st.session_state.img,
                    epsilon=PGD_EPSILON,
                    alpha=PGD_ALPHA,
                    iters=PGD_ITERS,
                    dataset=FGSM_DATASET_KEY,
                )
            elif attack_type == "Boundary":
                try:
                    adv_img = boundary_attack(
                        model,
                        st.session_state.img,
                        dataset=FGSM_DATASET_KEY,
                    )
                except Exception as e:
                    st.error(f"Boundary Attack failed: {e}")
                    adv_img = None
            elif attack_type == "HopSkipJump":
                try:
                    adv_img = hopskipjump_attack(
                        model,
                        st.session_state.img,
                        dataset=FGSM_DATASET_KEY,
                    )
                except Exception as e:
                    st.error(f"HopSkipJump Attack failed: {e}")
                    adv_img = None
            if adv_img is not None:
                adv_pred_class, adv_conf = predict_image(model, adv_img, dataset=FGSM_DATASET_KEY)
                st.session_state.adv_img = adv_img
                st.session_state.adv_pred = adv_pred_class
                st.session_state.adv_conf = adv_conf

        if st.button("⏭️ Next Random Image"):
            img, label = get_sample()
            # Ensure image is the correct size for GTSRB
            if is_gtsrb and (img.shape[0] != 48 or img.shape[1] != 48):
                img = Image.fromarray((img * 255).astype("uint8"))
                img = img.resize((48, 48))
                img = np.array(img).astype("float32") / 255.0
            st.session_state.img = img
            st.session_state.label = label
            st.session_state.pred = None
            st.session_state.conf = None
            st.session_state.adv_img = None
            st.session_state.adv_pred = None
            st.session_state.adv_conf = None
            if "user_uploaded" in st.session_state:
                del st.session_state["user_uploaded"]
            st.rerun()
    col_pred, col_adv = st.columns(2, gap="large")
    with col_pred:
        st.markdown("### Original Image Prediction")
        if st.session_state.get("pred") is not None:
            st.success(f"**Model Prediction:** {CLASS_NAMES[st.session_state.pred]}")
            st.write(f"**Confidence:** {st.session_state.conf:.2f}")
        else:
            st.info("Click **Predict** to see the model's prediction.")
    with col_adv:
        st.markdown("### Adversarial Image Prediction")
        if st.session_state.get("adv_pred") is not None:
            st.warning(f"**Adversarial Prediction:** {CLASS_NAMES[st.session_state.adv_pred]}")
            st.write(f"**Confidence:** {st.session_state.adv_conf:.2f}")
            if st.session_state.pred != st.session_state.adv_pred:
                st.error(
                    f"Prediction changed: {CLASS_NAMES[st.session_state.pred]} → {CLASS_NAMES[st.session_state.adv_pred]}"
                )
        else:
            st.info("Click **Generate Adversarial Example** to see the attack effect.")