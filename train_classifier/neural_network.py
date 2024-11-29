import torch.nn as nn
import torch
import yaml

def label_dict_from_config_file(relative_path):
    with open(relative_path,"r") as f:
       label_tag = yaml.full_load(f)["gestures"]
    return label_tag

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        list_label = label_dict_from_config_file("../hand_gesture.yaml")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, len(list_label))
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x

    def predict(self, x, threshold=0.8):
        logits = self.forward(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        values, chosen_ind = torch.max(softmax_prob, dim=1)
        return torch.where(values >= threshold, chosen_ind, torch.tensor(-1, device=x.device))

    def predict_with_known_classes(self, x):
        logits = self.forward(x)
        softmax_prob = nn.Softmax(dim=1)(logits)
        chosen_ind = torch.argmax(softmax_prob,dim=1)
        return chosen_ind

    def score(self, logits):
        return -torch.amax(logits, dim=1)