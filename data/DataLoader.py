from torch.utils.data import Dataset
import os   
import torch

# Dataset class for graph data
class MyGraphDataset(Dataset):
    def __init__(self, path):
        """
        Initialize the dataset with the specified path.

        Args:
            path (str): Path to the folder containing graph data files.
        """
        self.path = path

    def __len__(self):
        """
        Return the number of examples in the dataset.

        Returns:
            int: Number of graph data files in the specified folder.
        """
        # Change the current working directory to the dataset path
        os.chdir(self.path)
        # List all files in the folder
        file_list = os.listdir()
        # Return the number of files as the length of the dataset
        return len(file_list)

    def __getitem__(self, idx):
        """
        Get a specific example (graph data) from the dataset.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            torch_geometric.data.Data: Graph data loaded from the specified file.
        """
        # Construct the path to the file for the given index
        new_path = self.path + 'graph_data_' + str(idx) + '.pt'
        # Load the graph data from the file using torch.load
        data = torch.load(new_path)
        # Return the loaded graph data
        return data
