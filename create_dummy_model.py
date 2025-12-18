import torch
import torch.nn as nn

# Define the same model as main.py
class GestureMLP(nn.Module):
    def __init__(self, input_size=42, hidden_size=64, output_size=4):
        super(GestureMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model
model = GestureMLP()

# Save dummy model
torch.save(model.state_dict(), "gestures/gesture_model.pth")
print("✅ Dummy model saved successfully!")
