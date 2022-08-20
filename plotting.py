import numpy as np
import matplotlib.pyplot as plt
import time


def tensor_to_img(tensor):
    tensor = tensor.squeeze().permute(1, 2, 0)
    tensor = tensor.to("cpu").clone().detach()
    img = tensor.numpy()
    img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456,
                                                            0.406))
    img = img.clip(0, 1)
    return img


def imshow(tensor, ax, title=None):
    image = tensor_to_img(tensor)
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)


def plotting(output_img, style_img, content_img, dirname):
    counter = 3
    fig, axs = plt.subplots(1, counter)
    imshow(style_img, axs[0], title='Style Image')

    imshow(content_img, axs[1], title='Content Image')

    out_ax = axs[2] if counter > 1 else axs
    imshow(output_img, out_ax, title='Output Image')
    fname = "comparison.jpg"
    plt.savefig(f"data/{dirname}/{fname}")
    
    plt.clf()
    out_fname = "output.jpg"
    plt.imshow(tensor_to_img(output_img))
    plt.savefig(f"data/{dirname}/{out_fname}")
    
