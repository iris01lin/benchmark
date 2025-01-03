import tensorflow as tf
import tensorflow_datasets as tfds
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a simple CNN model on MNIST.")
    parser.add_argument('--num_gpus', type=int, default=0, help="Total number of GPUs to use.")
    parser.add_argument('--gpu_memory_limit', type=float, default=1024, help="Limit GPU memory usage(MB).")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--model_save_path', type=str, default='/var/tf_mnist/mnist_model.h5', help="Path to save the trained model.")
    return parser.parse_args()

def load_and_preprocess_data(batch_size, image_size):
    """Load and preprocess MNIST dataset."""
    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Shuffle the dataset
    train_dataset = train_dataset.shuffle(10000, seed=42)

    # Preprocessing function
    def preprocess_image(image, label):
        image = tf.image.resize(image, (image_size, image_size))  # Resize image
        image = tf.cast(image, tf.float32) / 255.0  # Normalize the image
        return image, label

    # Apply preprocessing and batch the dataset
    train_dataset = train_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset, test_dataset

def build_model(image_size):
    """Build a simple 2-layer CNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for MNIST digits
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def setup_device(num_gpus, gpu_memory_limit):
    """Set up device configuration for training."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if num_gpus > 0 and gpus:
        # Limit GPU memory growth to avoid over-allocation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            try:
                # Set the GPU memory fraction (e.g., 0.2 for 20% of the GPU memory)
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]  # Set limit in MB
                )
            except RuntimeError as e:
                print(e)
        
        # Specify the number of GPUs to use (if available)
        if num_gpus > len(gpus):
            print(f"Warning: Only {len(gpus)} GPUs are available, using all of them.")
        return tf.distribute.MirroredStrategy()  # Use MirroredStrategy for multi-GPU training
    else:
        # If no GPUs, set the CPU device
        return tf.distribute.get_strategy()  # Default strategy for single device (CPU)

def train_model(model, train_dataset, test_dataset, epochs):
    """Train the model."""
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=test_dataset
    )

def save_model(model, save_path):
    """Save the trained model to a specified path."""
    model.save(save_path)
    print(f"Model saved to {save_path}")

def main():
    # Parse arguments
    args = parse_args()

    # Setup device
    strategy = setup_device(args.num_gpus, args.gpu_memory_limit)
    
    # Use the strategy for distributing the training across devices (if any)
    with strategy.scope():
        # Load and preprocess the dataset
        train_dataset, test_dataset = load_and_preprocess_data(args.batch_size, 28)

        # Build and compile the model
        model = build_model(28)

        # Train the model
        train_model(model, train_dataset, test_dataset, args.epochs)

        # Save the model
        save_model(model, args.model_save_path)

if __name__ == "__main__":
    main()
