import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Define the joint embedding model
class JointEmbeddingModel(nn.Module):
    def __init__(self, input_size, embedding_size, output_size):
        super(JointEmbeddingModel, self).__init__()
        
        # Define the shared layers
        self.shared_layer_1 = nn.Linear(input_size, 128)
        self.shared_layer_2 = nn.Linear(128, 64)
        self.shared_layer_3 = nn.Linear(64, embedding_size)
        
        # Define the individual layers
        self.individual_layer_1 = nn.Linear(embedding_size * 2, 16)
        self.individual_layer_2 = nn.Linear(embedding_size * 2, 16)
        
        # Define the output layer
        self.output_layer = nn.Linear(16, output_size)
        
    def forward(self, input_1, input_2):
        # Pass the inputs through the shared layers
        embedding_1 = self.shared_layer_3(self.shared_layer_2(self.shared_layer_1(input_1)))
        embedding_2 = self.shared_layer_3(self.shared_layer_2(self.shared_layer_1(input_2)))
        
        # Concatenate the embeddings
        concatenated_embedding = torch.cat((embedding_1, embedding_2), dim=1)
        
        # Pass the concatenated embeddings through the individual layers
        individual_output_1 = self.individual_layer_1(concatenated_embedding)
        individual_output_2 = self.individual_layer_2(concatenated_embedding)
        
        # Pass the individual outputs through the output layer
        output_1 = self.output_layer(individual_output_1)
        output_2 = self.output_layer(individual_output_2)
        
        return output_1, output_2

# Define the dataset class
class UnitTestDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the training and testing data
X = np.random.rand(1000, 50)
y = np.random.randint(2, size=(1000, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the dataset and dataloaders
train_dataset = UnitTestDataset(X_train, y_train)
test_dataset = UnitTestDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model, loss function, and optimizer
model = JointEmbeddingModel(input_size=50, embedding_size=32, output_size=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        output_1, output_2 = model(inputs[:, :25], inputs[:, 25:])
        loss = criterion(output_1, targets) + criterion(output_2, targets)
        loss.backward()
