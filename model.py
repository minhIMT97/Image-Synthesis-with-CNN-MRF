from __future__ import print_function
import torch.nn as nn
import torchvision.models as models
import numpy as np
from mylibs import ContentLoss, StyleLoss, TVLoss, ContentFidelity

def GC(x,s):
    # GE
    GC = 0
    for i in range(3):
      hist_x = np.histogram(x[:,:,i], bins = 20)[0]
      hist_s = np.histogram(s[:,:,i], bins = 20)[0]
      GC += hist_x*hist_x/(np.linalg.norm(hist_x)*np.linalg.norm(hist_x)) 
    GC /= 3
    return np.sum(GC)

class CNNMRF(nn.Module):
    def __init__(self, style_image, content_image, model, device, content_weight, style_weight, tv_weight, gpu_chunck_size=256, mrf_style_stride=2,
                 mrf_synthesis_stride=2):
        super(CNNMRF, self).__init__()
        # fine tune alpha_content to interpolate between the content and the style
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.patch_size = 3
        self.device = device
        self.gpu_chunck_size = gpu_chunck_size
        self.mrf_style_stride = mrf_style_stride
        self.mrf_synthesis_stride = mrf_synthesis_stride
        # self.style_layers = [7,11] #  # vgg19 [11,20], resnet_teacher_note [5, slice(None, 3)], resnet_style [11,13]
        # self.style_layers_resnet = [1,5]
        # self.content_layers = [15] #  # vgg19 [22], resnet content [19]
        # self.content_layers_resnet = [2]
        if model == 'vgg':
          self.style_layers = [11,20] # vgg19 [11,20] resnet_teacher_note [5, slice(None, 3)]
          self.content_layers = [22] # vgg19 [22]
          self.model, self.content_losses, self.style_losses, self.tv_loss = \
              self.get_model_and_losses(style_image=style_image, content_image=content_image)
          
        elif model == 'resnet':
          self.style_layers = [7,11]
          self.content_layers = [15]
          self.model, self.content_losses, self.style_losses, self.tv_loss = \
              self.get_model_and_losses_resnet(style_image=style_image, content_image=content_image)

    def forward(self, synthesis):
        """
        calculate loss and return loss
        :param synthesis: synthesis image
        :return:
        """
        self.model(synthesis)
        style_score = 0
        content_score = 0
        tv_score = self.tv_loss.loss
        CF_score = 0
        SF_score = 0

        # calculate style loss
        for sl in self.style_losses:
            style_score += sl.loss
            SF_score += sl.fidelity # local pattern fidelity


        # calculate content loss
        for cl in self.content_losses:
            content_score += cl.loss
            CF_score += cl.fidelity # content fidelity


        # calculate final loss
        scale = 1
        loss = scale*(self.style_weight * style_score + self.content_weight * content_score + self.tv_weight * tv_score)
        CF_score_final = CF_score/len(self.content_losses)
        SF_score_final = SF_score/len(self.style_losses)
        return loss, CF_score_final, SF_score_final

    def update_style_and_content_image(self, style_image, content_image):
        """
        update the target of style loss layer and content loss layer
        :param style_image:
        :param content_image:
        :return:
        """
        # update the target of style loss layer
        x = style_image.clone()
        next_style_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TVLoss) or isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss): # or isinstance(layer, ContentFidelity):
                continue
            if next_style_idx >= len(self.style_losses):
                break
            x = layer(x)
            if i in self.style_layers:
                # extract feature of style image in vgg19 as style loss target
                self.style_losses[next_style_idx].update(x)
                next_style_idx += 1
            i += 1

        # update the target of content loss layer
        x = content_image.clone()
        next_content_idx = 0
        i = 0
        for layer in self.model:
            if isinstance(layer, TVLoss) or isinstance(layer, ContentLoss) or isinstance(layer, StyleLoss): # or isinstance(layer, ContentFidelity):
                continue
            if next_content_idx >= len(self.content_losses):
                break
            x = layer(x)
            if i in self.content_layers:
                # extract feature of content image in vgg19 as content loss target
                self.content_losses[next_content_idx].update(x)
                next_content_idx += 1
            i += 1
        
    # def get_metrics(self, synthesis):
      

    def get_model_and_losses(self, style_image, content_image):
        """
        create network model by intermediate layer of vgg19 and some customized layer(style loss, content loss and tv loss)
        :param style_image:
        :param content_image:
        :return:
        """
        vgg = models.vgg19(pretrained=True).to(self.device)
        model = nn.Sequential()
        content_losses = []
        style_losses = []
        # add tv loss layer
        tv_loss = TVLoss()
        model.add_module('tv_loss', tv_loss)

        next_content_idx = 0
        next_style_idx = 0

        for i in range(len(vgg.features)):
            if next_content_idx >= len(self.content_layers) and next_style_idx >= len(self.style_layers):
                break
            # add layer of vgg19
            layer = vgg.features[i]
            name = str(i)
            print(name, layer)
            model.add_module(name, layer)
            print('model: ', model)

            # add content loss layer
            if i in self.content_layers:
                target = model(content_image).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(next_content_idx), content_loss)
                content_losses.append(content_loss)
                next_content_idx += 1

            # add style loss layer
            if i in self.style_layers:
                target_feature = model(style_image).detach()
                style_loss = StyleLoss(target_feature, patch_size=self.patch_size, mrf_style_stride=self.mrf_style_stride,
                                       mrf_synthesis_stride=self.mrf_synthesis_stride, gpu_chunck_size=self.gpu_chunck_size, device=self.device)

                model.add_module("style_loss_{}".format(next_style_idx), style_loss)
                style_losses.append(style_loss)
                next_style_idx += 1

        return model, content_losses, style_losses, tv_loss

    def get_model_and_losses_resnet(self, style_image, content_image):
        """
        create network model by intermediate layer of resnet and some customized layer(style loss, content loss and tv loss)
        :param style_image:
        :param content_image:
        :return:
        """
        vgg = models.resnet34(pretrained=True).to(self.device)
        model = nn.Sequential()
        content_losses = []
        style_losses = []
        content_fidelities = []
        # add tv loss layer
        tv_loss = TVLoss()
        model.add_module('tv_loss', tv_loss)
        # print(vgg._modules['layer1']._modules['2']._modules.keys())
        next_content_idx = 0
        next_style_idx = 0
        idx = 4
        for i in range(len(list(vgg.children())[:])):
            if next_content_idx >= len(self.content_layers) and next_style_idx >= len(self.style_layers):
                break
            # add layer of ResNet
            layer = list(vgg.children())[:][i]
            name = str(i)
            
            if i < 4:
              print(name, layer)
              model.add_module(name, layer)
              print('model: ', model)

              # add content loss layer
              if i in self.content_layers:
                  target = model(content_image).detach()
                  content_loss = ContentLoss(target)
                  model.add_module("content_loss_{}".format(next_content_idx), content_loss)
                  content_losses.append(content_loss)
                  
                  next_content_idx += 1

              # add style loss layer
              if i in self.style_layers:
                  target_feature = model(style_image).detach()
                  style_loss = StyleLoss(target_feature, patch_size=self.patch_size, mrf_style_stride=self.mrf_style_stride,
                                          mrf_synthesis_stride=self.mrf_synthesis_stride, gpu_chunck_size=self.gpu_chunck_size, device=self.device)

                  model.add_module("style_loss_{}".format(next_style_idx), style_loss)
                  style_losses.append(style_loss)
                  next_style_idx += 1              
            else:
              for j in range(len(list(layer.children()))):
                sublayer = list(layer.children())[j]
                subname = str(j+idx)
                model.add_module(subname, sublayer)
                print('model: ', model)

                # add content loss layer
                if (j+idx) in self.content_layers:
                    target = model(content_image).detach()

                    content_loss = ContentLoss(target)
                    model.add_module("content_loss_{}".format(next_content_idx), content_loss)
                    content_losses.append(content_loss)
                    
                    next_content_idx += 1

                # add style loss layer
                if (j+idx) in self.style_layers:
                    target_feature = model(style_image).detach()
                    style_loss = StyleLoss(target_feature, patch_size=self.patch_size, mrf_style_stride=self.mrf_style_stride,
                                            mrf_synthesis_stride=self.mrf_synthesis_stride, gpu_chunck_size=self.gpu_chunck_size, device=self.device)

                    model.add_module("style_loss_{}".format(next_style_idx), style_loss)
                    style_losses.append(style_loss)
                    next_style_idx += 1

                if j == len(list(layer.children())) - 1:
                    idx = j+idx + 1

        return model, content_losses, style_losses, tv_loss
