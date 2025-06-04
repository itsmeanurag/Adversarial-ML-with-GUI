import tensorflow as tf

model_files = [
    "models/mnist_model.h5",
    "models/cifar100_model.h5",
    "models/gtsrb_model.h5",
    # Add more model file paths as needed
]

for model_path in model_files:
    print(f"Checking {model_path} ...")
    try:
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(model_path)
        print(f"✅ {model_path} loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}\n")