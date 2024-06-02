import numpy as np


def __adjust_array(arr, target_length=150):
    # Get the current number of rows in the array
    current_length = arr.shape[0]

    if current_length > target_length:
        # Truncate the array if it's longer than the target
        adjusted_array = arr[:target_length]
    elif current_length < target_length:
        # Calculate the number of rows to pad
        rows_to_add = target_length - current_length
        # Get the number of columns in the array
        num_columns = arr.shape[1]
        # Create an array of zeros to pad
        padding = np.zeros((rows_to_add, num_columns), dtype=arr.dtype)
        # Append the padding to the original array
        adjusted_array = np.vstack((arr, padding))
    else:
        # If the length is already target_length, return the original array
        adjusted_array = arr

    return adjusted_array


def get_preds(model_output):
    return np.argmax(model_output, axis=1)
