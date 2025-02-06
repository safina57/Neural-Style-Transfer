import torch
from torchvision import transforms
from PIL import Image

import torch.optim as optim
import matplotlib.pyplot as plt

class NST_Model:
    def __init__(self, vgg, alpha : float = 5e1, beta : float = 2e2, weight : float = 2e-2, imsize : int = 512):
        self.vgg = vgg
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.weight = torch.tensor(weight)
        self.imsize = imsize
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg.to(self.device)

    def reshape_tensor(self, input: torch.tensor) -> torch.tensor:
        """
        Reshape tensor to 2D tensor
        
        Args:
            - input: tensor to be reshaped

        Returns:
            - reshaped tensor
        """
        return input.view(input.size(1), -1)

    def content_loss(self, content: torch.tensor, generated: torch.tensor) -> torch.tensor:
        """
        Compute content loss

        Args:
            - content: content tensor
            - generated: generated tensor

        Returns:
            - content loss
        """
        content = self.reshape_tensor(content)
        generated = self.reshape_tensor(generated)
        return torch.mean((content - generated) ** 2)

    def gram_matrix(self, input: torch.tensor) -> torch.tensor:
        """
        Compute gram matrix

        Args:
            - input: tensor to compute gram matrix

        Returns:
            - gram matrix
        """
        return torch.mm(input, input.t())

    def style_layer_loss(self, style: torch.tensor, generated: torch.tensor) -> torch.tensor:
        """
        Compute style layer loss

        Args:
            - style: style tensor
            - generated: generated tensor

        Returns:
            - style layer loss
        """
        style = self.reshape_tensor(style)
        generated = self.reshape_tensor(generated)
        style_gram = self.gram_matrix(style)
        generated_gram = self.gram_matrix(generated)
        return torch.mean((style_gram - generated_gram) ** 2) / generated.numel()

    def style_loss(self, style_layers: list, generated_layers: list) -> torch.tensor:
        """
        Compute style loss

        Args:
            - style_layers: list of style layers
            - generated_layers: list of generated layers

        Returns:
            - style loss
        """
        J_style = torch.tensor(0.0, device=self.device)
        for i in range(len(style_layers)):
            J_style += self.weight * self.style_layer_loss(style_layers[i], generated_layers[i])
        return J_style

    def total_loss(self, J_content: torch.tensor, J_style: torch.tensor) -> torch.tensor:
        """
        Compute total loss

        Args:
            - J_content: content loss
            - J_style: style loss

        Returns:
            - total loss
        """
        return self.alpha * J_content + self.beta * J_style

    def image_loader(self, image_name: str) -> torch.tensor:
        """
        Load image and preprocess it

        Args:
            - image_name: image name

        Returns:
            - preprocessed image
        """
        image = Image.open(image_name)
        loader = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = loader(image).unsqueeze(0)
        return image.to(self.device)

    def imshow(self, tensor: torch.tensor, title: str = None):
        """
        Display tensor as image

        Args:
            - tensor: tensor to display
            - title: title of the image
        """
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        unloader = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image = unloader(image)
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        plt.axis("off")
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def generate_noise_image(self, content: torch.tensor) -> torch.tensor:
        """
        Generate noise image

        Args:
            - content: content image

        Returns:
            - noise image
        """
        torch.manual_seed(17)
        noise = torch.rand_like(content)
        return noise

    def get_optimizer(self, input: torch.tensor) -> optim.LBFGS:
        """
        Get optimizer

        Args:
            - input: input tensor

        Returns:
            - optimizer
        """
        optimizer = optim.LBFGS([input.requires_grad_()])
        return optimizer

    def train(self, content: torch.tensor, style: torch.tensor, generated: torch.tensor, epochs: int = 1000):
        """
        Train the model

        Args:
            - content: content image
            - style: style image
            - generated: generated image
            - epochs: number of epochs
        """
        content = content.to(self.device)
        style = style.to(self.device)
        generated = generated.to(self.device)

        with torch.no_grad():
            content_features = self.vgg.forward(content, is_style=False)
            style_features = self.vgg.forward(style, is_style=True)

        optimizer = self.get_optimizer(generated)

        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                generated_features = self.vgg.forward(generated, is_style=False)
                J_content = self.content_loss(content_features[0], generated_features[0])
                generated_style_features = self.vgg.forward(generated, is_style=True)
                J_style = self.style_loss(style_features, generated_style_features)
                J_total = self.total_loss(J_content, J_style)
                J_total.backward()
                return J_total

            if epoch % 10 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    generated_features = self.vgg.forward(generated, is_style=False)
                    J_content = self.content_loss(content_features[0], generated_features[0])
                    generated_style_features = self.vgg.forward(generated, is_style=True)
                    J_style = self.style_loss(style_features, generated_style_features)
                    J_total = self.total_loss(J_content, J_style)
                    print(f'Epoch {epoch} - Total loss: {J_total.item()}, Content loss: {J_content.item()}, Style loss: {J_style.item()}')
                    self.imshow(generated, title=f'Epoch {epoch}')

            optimizer.step(closure)

        return generated

    def save_and_display_image(self, generated: torch.tensor, name: str):
        """
        Save and display image

        Args:
            - generated: generated image
            - name: name of the image
        """
        image = generated.cpu().clone()
        image = image.squeeze(0)
        unloader = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        image = unloader(image)
        image = torch.clamp(image, 0, 1)
        image = transforms.ToPILImage()(image)
        plt.axis("off")
        image.save(f"data/generated/{name}.jpg")
        plt.imshow(image)
