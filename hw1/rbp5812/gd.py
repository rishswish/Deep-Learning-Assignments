import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import VGG13_BN_Weights, vgg13_bn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def save_img(image, path):
    # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
    image = image[0].permute(1, 2, 0)
    image = image.clamp(min=0, max=1)
    image = (image * 255).cpu().detach().numpy().astype(np.uint8)
    # opencv expects BGR (and not RGB) format
    cv.imwrite(path, image[:, :, ::-1])


def main():
    model = vgg13_bn(VGG13_BN_Weights.IMAGENET1K_V1).to(DEVICE)
    print(model)
    for label in [0, 12, 954]:
        image = torch.randn(1, 224, 224, 3).to(DEVICE)
        image = (image * 8 + 128) / 255  # background color = 128,128,128
        image = image.permute(0, 3, 1, 2)
        image.requires_grad_()
        image = gradient_descent(image, model, lambda tensor: tensor[0, label].mean(),)
        save_img(image, f"./img_{label}.jpg")
        out = model(image)
        print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")


# DO NOT CHANGE ANY OTHER FUNCTIONS ABOVE THIS LINE FOR THE FINAL SUBMISSION


def normalize_and_jitter(img, step=32):
    dx, dy = np.random.randint(-step, step - 1, 2)
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(
        img.roll(dx, -1).roll(dy, -2)
    )


def blur_tensor(img, ksize=3):
    """Apply Gaussian blur to image tensor."""
    b, c, h, w = img.shape
    img_np = img.detach().cpu().numpy().transpose(0, 2, 3, 1)
    blurred = []
    for i in range(b):
        blurred.append(cv.GaussianBlur(img_np[i], (ksize, ksize), 0))
    blurred = np.stack(blurred, axis=0)
    return torch.from_numpy(blurred.transpose(0, 3, 1, 2)).to(img.device)


def blur_gradients(grad, ksize=3):
    """Apply Gaussian blur to gradients."""
    b, c, h, w = grad.shape
    grad_np = grad.detach().cpu().numpy().transpose(0, 2, 3, 1)
    blurred = []
    for i in range(b):
        blurred.append(cv.GaussianBlur(grad_np[i], (ksize, ksize), 0))
    blurred = np.stack(blurred, axis=0)
    return torch.from_numpy(blurred.transpose(0, 3, 1, 2)).to(grad.device)


def gradient_descent(
    input,
    model,
    loss_fn,
    iterations=128,
    learning_rate=0.01,
    weight_decay=1e-4,
    grad_clip=5.0,
):
    """
    Perform gradient ascent on the input image to maximize a target class score.
    """

    for i in tqdm(range(iterations), desc="Gradient Descent"):
        # Normalize + jitter before passing through model
        normed = normalize_and_jitter(input.clone(), step=4)

        # Forward pass
        out = model(normed)

        # Compute loss
        loss = loss_fn(out)

        # Zero grads
        for p in model.parameters():
            p.grad = None
        if input.grad is not None:
            input.grad.zero_()

        # Backward
        loss.backward()

        with torch.no_grad():
            grad = input.grad

            # Blur gradients to reduce noise
            grad = blur_gradients(grad, ksize=3)

            # Clip gradients
            if grad_clip is not None:
                grad = torch.clamp(grad, -grad_clip, grad_clip)

            # Update image (gradient ascent)
            input.data.add_(grad, alpha=learning_rate)

            # Weight decay to reduce explosion
            input.data.mul_(1 - weight_decay)

            # Occasionally blur the image itself
            if i % 10 == 0:
                input.data = blur_tensor(input.data, ksize=3)

            # Keep image in valid range
            input.data.clamp_(0, 1)

    return input


# ---------------------------------------------------------------
# Extra Credit: Return Activations from Intermediate Layers
# ---------------------------------------------------------------

def forward_and_return_activation(model, input, module):
    """
    This function is for the extra credit. You may safely ignore it.
    Given a module in the middle of the model (like `model.features[20]`),
    it will return the intermediate activations.
    """
    features = []

    def hook(model, input, output):
        features.append(output)

    handle = module.register_forward_hook(hook)
    model(input)
    handle.remove()

    return features[0]


if __name__ == "__main__":
    main()