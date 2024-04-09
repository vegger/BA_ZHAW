from torch.utils.data import Dataset

class NoStitchrDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: The data you want to load. This could be a list, a file path, etc.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data

    def __len__(self):
        # Return the size of your dataset
        return len(self.data)

    def __getitem__(self, index):
        # Fetch the data point at the given index
        data_point = self.data[index]

        if self.transform:
            # Apply transformations to the data point if any
            data_point = self.transform(data_point)

        return data_point
