import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader

class SequenceDataGenerator(Dataset):
    def __init__(self, data, feature_cols, target_cols, seq_len=95, batch_size=64):
        self.groups = list(data.groupby('MMSI'))
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.indices = self._create_indices()

    def _create_indices(self):
        indices = []
        for group_idx, (_, group) in enumerate(self.groups):
            for i in range(len(group) - self.seq_len):
                indices.append((group_idx, i))
        return indices

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def extra_new(self, X, Y):
        enu_coordinates = []
        for i in range(len(X)):
            latA, lonA = X[i][-1][:2][0], X[i][-1][:2][1]  # Reference point A
            latB, lonB = Y[i][0], Y[i][1]  # Point B
            #print(X[i][0][:2])
            transformer_to_enu = Transformer.from_crs("epsg:4326", f"+proj=aeqd +lat_0={latA} +lon_0={lonA} +datum=WGS84", always_xy=True)
            transformer_to_global = Transformer.from_crs(f"+proj=aeqd +lat_0={latA} +lon_0={lonA} +datum=WGS84", "epsg:4326", always_xy=True)
            x_enu, y_enu = transformer_to_enu.transform(lonB, latB)
            enu_coordinates.append((x_enu, y_enu))
        return torch.tensor(enu_coordinates, dtype=torch.float32)


    def __getitem__(self, idx):
        X_batch = []
        Y_batch = []
        
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        
        for group_idx, data_idx in self.indices[start_idx:end_idx]:
            group = self.groups[group_idx][1]
            X = group[self.feature_cols].iloc[data_idx:data_idx + self.seq_len, :]#.values
            Y = group[self.target_cols].iloc[data_idx + self.seq_len]#.values
            #print(X.isnull().sum().sum(), Y.isnull().sum())
            #if (X.isnull().sum().sum()==0) and (Y.isnull().sum()==0) :
            X_batch.append(X.values)
            Y_batch.append(Y.values)
        #print(len(X_batch))
        
        Y_batch = self.extra_new(X_batch, Y_batch)
        return torch.tensor(X_batch, dtype=torch.float32), torch.tensor(Y_batch, dtype=torch.float32)

# Usage

feature_cols = data.columns[2:]
target_cols = data.columns[2:4]

train_generator = SequenceDataGenerator(train, feature_cols, target_cols, seq_len=45)
train_loader = DataLoader(train_generator, batch_size=1, shuffle=True)

test_generator = SequenceDataGenerator(test, feature_cols, target_cols, seq_len=45, batch_size=256)
test_loader = DataLoader(test_generator, batch_size=1, shuffle=True)

# Requesting batches of 64 data rows sequentially
for X, Y in train_loader:
    print(X.shape, Y.shape)
    break
