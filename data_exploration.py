from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds

# Name of the dataset inside TensorFlow Datasets
DATASET_NAME = "plant_village"

# Where TFDS stores the downloaded data (kept inside the git-ignored dataset/ folder)
DATA_DIR = Path("dataset/tfds")

# Folder where we save charts, CSV, and sample images
OUTPUTS_DIR = Path("outputs")


def load_dataset():
    """Download (first time) and load the PlantVillage dataset from TFDS.

    Returns the tf.data.Dataset (train split) and its info object.
    """
    print("Loading", DATASET_NAME, "from TensorFlow Datasets...")
    print(
        "(first run downloads ~800 MB into", str(DATA_DIR) + ", later runs are cached)"
    )
    ds, info = tfds.load(
        DATASET_NAME,
        split="train",  # PlantVillage only has a single 'train' split
        as_supervised=True,  # gives (image, label) pairs
        data_dir=str(DATA_DIR),
        with_info=True,  # also return dataset metadata
    )

    return ds, info


def get_class_names(info):
    """Return the list of class names and print a short summary."""
    class_names = info.features["label"].names
    print("\nTotal classes:", len(class_names))
    print("Total images:", info.splits["train"].num_examples)
    print("First 10 classes:")
    for name in class_names[:10]:
        print("-", name)
    return class_names


def count_images(class_names):
    """Count images per class and return a pandas DataFrame.

    We load the data again with image decoding skipped, so we only read the
    labels. This makes counting all 54k images fast.
    """
    print("\nCounting images per class...")
    count_ds = tfds.load(
        DATASET_NAME,
        split="train",
        data_dir=str(DATA_DIR),
        # Skip decoding the image bytes; we only need the label here
        decoders={"image": tfds.decode.SkipDecoding()},
    )

    # Start every class at 0, then add up the labels we see
    counts = [0] * len(class_names)
    for example in tfds.as_numpy(count_ds):
        counts[int(example["label"])] += 1

    df = pd.DataFrame({"class_name": class_names, "image_count": counts})
    print("\nImage count per class (first rows):")
    print(df.head())
    print("\nTotal images:", df["image_count"].sum())

    # Save the counts as a CSV file in outputs/
    csv_path = OUTPUTS_DIR / "image_counts.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    return df


def plot_class_distribution(df):
    """Save a simple bar chart of image count per class."""
    plt.figure(figsize=(14, 6))
    plt.bar(df["class_name"], df["image_count"])
    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Image Count")
    # Rotate labels so long class names stay readable
    plt.xticks(rotation=90)
    plt.tight_layout()

    chart_path = OUTPUTS_DIR / "class_distribution.png"
    plt.savefig(chart_path)
    plt.close()
    print("Saved:", chart_path)


def save_sample_images(ds, class_names, num_samples=4):
    """Save a few sample images from the dataset with their class names."""
    plt.figure(figsize=(14, 4))
    # Take a few (image, label) pairs from the dataset
    for i, (image, label) in enumerate(tfds.as_numpy(ds.take(num_samples))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(class_names[int(label)], fontsize=9)
        plt.axis("off")
    plt.tight_layout()

    samples_path = OUTPUTS_DIR / "sample_images.png"
    plt.savefig(samples_path)
    plt.close()
    print("Saved:", samples_path)


def check_image_sizes(ds):
    """Print the size of a few images.

    Note: images may have different sizes. CNN models need all input images
    to be the same size, so later we will resize every image (for example to
    224x224) during preprocessing before training.
    """
    print("\nSample image sizes (height x width x channels):")
    for image, _ in tfds.as_numpy(ds.take(5)):
        print(image.shape)


def count_healthy_vs_diseased(df):
    """Count healthy vs diseased images using the class names.

    Class names that contain "healthy" are healthy leaves.
    All other classes are treated as diseased (binary classification idea).
    """
    healthy_count = 0
    diseased_count = 0
    for row in df.itertuples():
        if "healthy" in row.class_name.lower():
            healthy_count += row.image_count
        else:
            diseased_count += row.image_count

    print("\nHealthy images:", healthy_count)
    print("Diseased images:", diseased_count)


def main():
    print("Dataset Exploration - PlantVillage\n")

    # Make sure the outputs/ folder exists
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Load the dataset (downloads on first run)
    ds, info = load_dataset()

    # Run the exploration steps
    class_names = get_class_names(info)
    df = count_images(class_names)
    plot_class_distribution(df)
    save_sample_images(ds, class_names)
    check_image_sizes(ds)
    count_healthy_vs_diseased(df)

    # Short summary
    print("\nSummary:")
    print("- Dataset loaded from TensorFlow Datasets (plant_village)")
    print("- Class names listed")
    print("- Images counted per class (outputs/image_counts.csv)")
    print("- Class distribution chart saved (outputs/class_distribution.png)")
    print("- Sample images saved (outputs/sample_images.png)")
    print("- Healthy vs diseased count checked")
    print(
        "- Next step: start basic preprocessing and prepare train/validation/test split"
    )


if __name__ == "__main__":
    main()
