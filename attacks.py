import torch
import torch.nn as nn

class FGSM:
    def __init__(self, model, eps=0.031):
        self.model = model
        self.eps = eps
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, images, labels):
        images = images.clone().detach().to(labels.device)
        images.requires_grad = True
        
        outputs = self.model(images)
        loss = self.loss_fn(outputs, labels)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Calculate FGSM perturbation
        data_grad = images.grad.data
        perturbed_image = images + self.eps * data_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1) # Clipping to valid image range
        
        return perturbed_image

class PGD:
    def __init__(self, model, eps=0.031, alpha=0.01, steps=10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, images, labels):
        images_orig = images.clone().detach()
        images = images.clone().detach().to(labels.device)
        
        # Iterative PGD attack
        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Step in direction of gradient
            adv_images = images + self.alpha * images.grad.data.sign()
            
            # Projection back to L-infinity ball around original image
            eta = torch.clamp(adv_images - images_orig, min=-self.eps, max=self.eps)
            images = torch.clamp(images_orig + eta, min=0, max=1).detach_()
            
        return images