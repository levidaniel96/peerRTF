from torch.utils.data import Dataset
import os   
import torch
import pandas as pd
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
        new_path = self.path + 'graph_data.pt'
        # Load the graph data from the file using torch.load
        data = torch.load(new_path)
        # Return the loaded graph data
        return data


def print_DNSMOS_results(csv_path):
    df = pd.read_csv(csv_path+'DNSMOS_epoc_100.csv')
    ## add gcn from csv
    df['type']=df['filename'].apply(lambda x: x.split('/')[-1].split('.')[0])
    ## print the results
    print('DNSMOS results:')
    for i in range(len(df)):
        print(df['type'][i], '{:.2f}'.format(df['P808_MOS'][i]))

    
  