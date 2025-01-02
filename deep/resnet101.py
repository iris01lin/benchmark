import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_datasets as tfds
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Cats vs Dogs model using ResNet101.")
    parser.add_argument('--num_gpus', type=int, default=0, help="Total number of GPUs to use.")
    parser.add_argument('--gpu_memory_limit', type=float, default=0.2, help="Fraction of GPU memory to use.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--image_size', type=int, default=224, help="Input image size for ResNet101.")
    parser.add_argument('--model_save_path', type=str, default='/var/resnet/cats_vs_dogs_model.h5', help="Path to save the trained model.")
    return parser.parse_args()

def load_and_preprocess_data(batch_size, image_size):
    """Load and preprocess Cats vs Dogs dataset."""
    dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)
    train_dataset = dataset['train']

    #Shuffle the dataset
    train_dataset = train_dataset.shuffle(10000, seed=42)

    # Calculate the number of validation samples based on the split percentage
    total_samples = len(list(train_dataset))
    val_samples = int(total_samples * 0.2)

    # Split the dataset into train and validation
    val_dataset = train_dataset.take(val_samples)
    train_dataset = train_dataset.skip(val_samples)

    def preprocess_image(image, label):
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_dataset = train_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset, val_dataset

def build_model(image_size):
    """Build and compile the ResNet101 model."""
    base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    base_model.trainable = False  # Freeze the base model

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

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
                tf.config.set_virtual_device_configuration(
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
        return tf.distribute.get_strategy()  # Default strategy for single device (CPU or 1 GPU)

def train_model(model, train_dataset, val_dataset, epochs):
    """Train the model."""
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset
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
        train_dataset, val_dataset = load_and_preprocess_data(args.batch_size, args.image_size)

        # Build and compile the model
        model = build_model(args.image_size)

        # Train the model
        train_model(model, train_dataset, val_dataset, args.epochs)

        # Save the model
        save_model(model, args.model_save_path)

if __name__ == "__main__":
    main()
