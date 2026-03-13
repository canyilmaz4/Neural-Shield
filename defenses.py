import torch
import torch.nn as nn

class AdversarialTrainer:
    def __init__(self, model, attacker):
        self.model = model
        self.attacker = attacker
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, images, labels, optimizer):
        # 1. Generate adversarial examples for the current batch
        self.model.eval()
        adv_images = self.attacker.perturb(images, labels)
        self.model.train()
        
        # 2. Compute loss on adversarial examples
        optimizer.zero_grad()
        outputs = self.model(adv_images)
        loss = self.loss_fn(outputs, labels)
        
        # 3. Optimize the model (standard backprop)
        loss.backward()
        optimizer.step()
        
        return loss.item()