from dataclasses import dataclass

'''
The code defines several data classes using the dataclass decorator
from the dataclasses module. Each data class represents a specific
configuration or set of parameters used in the model. 
'''

@dataclass
class Paths:
    train_path: str             # Path to the training data
    val_path: str               # Path to the validation data
    test_path: str              # Path to the test data    
    results_path: str           # Path to store the results
    modelData_path: str         # Path to store model data
    modelData_path_saved: str   # Path to store model data
    log_path: str               # Path to store log files
    csv_path: str               # Path to store csv files

@dataclass
class Params:
    Nl: int                     # Nl samples from the left side of the RTF
    Nr: int                     # Nr samples from the right side of the RTF
    Nl_in: int                  # Nl samples from the left side of the input RTF
    Nr_in: int                  # Nr samples from the right side of the input RTF
    M: int                      # Number of microphones
    feature_size: int           # RTFs 
    NFFT: int                   # FFT size
    n_hop: int                  # Hop size
    fs: int                     # Sampling frequency
    NUP: int                    # Number of frequency bins to keep 
    ref_mic: int                # Reference microphone
    both_tim_st: int            # start time stamp for both - noise and speech
    both_tim_fn: int            # end time stamp for both - noise and speech
    length_records: int         # Length of the recordings
@dataclass
class ModelParams:
    num_layers: int             # Number of layers in the model
    feature_size: int           # RTFs
    dropout: float              # Dropout rate
    activation: str             # Activation function
    activation_out: str         # Activation function for the output layer
    tanh_alpha: float           # Alpha parameter for the tanh activation function 
    batch_norm: bool            # Flag to use batch normalization
    train_mode: str             # Training mode
@dataclass
class Device:
    device_num: int             # Device number

@dataclass
class Loss:
    loss_mode: str              # Loss type
    loss_type: str              # Loss function
    loss_RTFs_shape: str        # Training mode
    loss_scale: float           # Loss scale
@dataclass
class Model_HP:
    train_size_spilt: float     # Training size split
    val_size_spilt: float       # Validation size split
    batchSize: int              # Batch size
    batchSize_test: int         # Batch size for test
    epochs: int                 # Epochs
    data_loader_shuffle: bool   # Flag to shuffle data in data loader
    test_loader_shuffle: bool   # Flag to shuffle data in test loader
    num_workers: int            # Number of workers
    model_type: str             # Model type
    num_hops: int               # Number of hops
@dataclass
class Optimizer:   
    optimizer: str              # Optimizer name
    learning_rate: float        # Learning rate
    weight_decay: float         # Weight decay
@dataclass
class flags:
    save_model: bool            # Flag to save the model
    save_RTFs: bool             # Flag to save the RTFs
    save_oracle_and_noisy: bool # Flag to save the oracle and noisy RTFs 
    save_mat: bool              # Flag to save the RTFs in .mat format
    load_weights: bool          # Flag to load weights
    test: bool                  # Flag to test the model   
@dataclass 
class massagePassingConfig:
    paths: Paths                # Paths configuration
    params: Params              # Parameters configuration
    modelParams: ModelParams    # Model Parameters configuration
    device: Device              # Device configuration
    loss: Loss                  # Loss configuration
    model_hp: Model_HP          # Model hyperparameters configuration
    optimizer: Optimizer        # Optimizer configuration
    flags: flags                # Flags configuration