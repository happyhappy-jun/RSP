import tensorflow as tf
import glob
import os

# Option 2: Use pattern matching to find all relevant files
tfrecord_files = glob.glob('/root/bridge/1.0.0/bridge_dataset-val.tfrecord-*')
print(tfrecord_files)

# Create a TFRecordDataset that reads from multiple files
dataset = tf.data.TFRecordDataset(tfrecord_files)


# Function to inspect contents
def inspect_multiple_tfrecords():
    count = 0
    for serialized_example in dataset:
        # Parse the example
        example = tf.train.Example.FromString(serialized_example.numpy())

        # Get the current file being processed (approximate method)
        current_file_index = min(count // 10, len(tfrecord_files) - 1)  # Assuming 10 examples per file
        current_file = tfrecord_files[current_file_index]

        # Print information about the example
        print(f"Example {count} (likely from {os.path.basename(current_file)}):")
        print(example)
        print("\n" + "-" * 50 + "\n")

        count += 1
        # Limit the number of examples to display
        if count >= 2:
            break

    print(f"Displayed {count} examples from the TFRecord files.")


# Execute the inspection
inspect_multiple_tfrecords()