from PIL import Image
from torchvision import transforms
import torch


def image_loader(image_name, imsize):
    image = Image.open(image_name)
    transform = transforms.Compose([
      transforms.Resize(imsize),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return transform(image)


def image_preprocess(image_name, imsize, iscuda=True):
    loader = image_loader(image_name, imsize)
    img = loader.reshape(1, 3, imsize[0], imsize[1])
    if iscuda:
        img = img.cuda()
    return img


def get_images(folder, imsize=(256, 256), iscuda=True):
    style_img = image_preprocess(f'data/{folder}/style.jpg', imsize, iscuda)
    content_img = image_preprocess(f'data/{folder}/content.jpg',
                                   imsize, iscuda)

    input_img = torch.randn(1, 3, imsize[0], imsize[1])
    if iscuda:
        input_img = input_img.cuda()
    return content_img, style_img, input_img
