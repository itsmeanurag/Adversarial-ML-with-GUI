import tensorflow as tf

def load_mnist_model(model_path="models/mnist_model.keras"):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

def load_cifar100_model(model_path="models/cifar100_model.keras"):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

def load_gtsrb_model(model_path="models/gtsrb_model.keras"):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)

def load_imagenet_model():
    from tensorflow.keras.applications import ResNet50
    tf.keras.backend.clear_session()
    return ResNet50(weights="imagenet")