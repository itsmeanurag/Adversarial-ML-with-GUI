import tensorflow as tf

print("TensorFlow version:", tf.__version__)

try:
    import keras
    print("Standalone keras version:", keras.__version__)
except ImportError:
    print("Standalone keras not installed (good for TF>=2.15)")