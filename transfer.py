import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim


class VGG19(torch.nn.Module):

    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.model = vgg19.features

    def forward(self, x):
        pools = (4, 9, 18)
        out = (
            self.model[:pools[0]](x),
            self.model[:pools[1]](x),
            self.model[:pools[2]](x)
        )
        return out


class Transferer:

    def __init__(self, content_img, style_img, input_img, iscuda=True):
        self.optimizer = optim.LBFGS([input_img.requires_grad_()])

        self.model = VGG19()
        if iscuda:
            self.model = self.model.cuda()
        self.content_img = content_img
        self.style_img = style_img
        self.input_img = input_img
        self.cont_eval = self.model(self.content_img)
        self.style_eval = self.model(self.style_img)

    def content_loss(self, content_features, input_features):
        return F.mse_loss(content_features, input_features)

    def calculate_gram_matrices(self, feature_tensor):
        a, b, c, d = feature_tensor.shape
        reshaped_features = feature_tensor.view(a, b, c*d)
        return torch.matmul(reshaped_features,
                            reshaped_features.transpose(1, 2))

    def style_loss(self, style_features, input_features):
        return F.mse_loss(self.calculate_gram_matrices(style_features),
                          self.calculate_gram_matrices(input_features))

    def fit(self, style_weight=1, content_weight=1000000, num_steps=300,
            cont_layers=range(3), style_layers=range(3)):
        for e in range(num_steps):
            print('epoch:', e)

            def closure():
                self.optimizer.zero_grad()
                out = self.model(self.input_img.clone())

                loss_cont = torch.stack([self.content_loss(
                    self.cont_eval[i],
                    out[i]) for i in cont_layers], dim=0).sum()
                loss_style = torch.stack([self.style_loss(
                    self.style_eval[i],
                    out[i]) for i in style_layers], dim=0).sum()

                loss = loss_cont*content_weight + loss_style*style_weight
                loss.backward(retain_graph=True)
                return loss
            self.optimizer.step(closure)

    def get_output(self, content=True, style=True):
        out = {'target': self.input_img}
        if content:
            out['content'] = self.content_img
        if style:
            out['style'] = self.style_img
        return out
