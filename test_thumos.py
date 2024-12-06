from datasets.thumos import ThumosDataset, ThumosClassificationDataset

import torch

if __name__ == "__main__":
    dataset = ThumosClassificationDataset(sampling_rate=8, n_videos=5)
    img, label = dataset[0]
    print(f"Number of frames: {len(dataset)}")
    n, h, w, c = dataset.imgs.shape
    print(f"n={n}, h={h}, w={w}, c={c}")

    # Count occurrences of each number from 0 to 20
    counts = torch.zeros(21, dtype=torch.int)  # Initialize a tensor to store counts for each number

    # Increment counts for each occurrence
    for i in range(21):
        counts[i] = (dataset.labels == i).sum()

    # Print the result
    for i, count in enumerate(counts):
        if i < len(ThumosDataset.categories):
            print(f"{ThumosDataset.categories[i]}: {count.item()} frames")
        else:
            print(f"No action: {count.item()} frames")

    # Print the result from the dataset attribute
    freqs = dataset.get_freqs()
    for i, freq in enumerate(freqs):
        if i < len(ThumosDataset.categories):
            print(f"{ThumosDataset.categories[i]}: {freq} frames")
        else:
            print(f"No action: {freq} frames")

    print("Success!")
