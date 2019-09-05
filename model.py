from fastai.callbacks import LossMetrics, F
from fastai.vision import imagenet_stats, get_transforms, ImageImageList, torch, requires_grad, children, \
    NormType, unet_learner, models
from torchvision.models import vgg16_bn
from loss import *


def get_model(test_path):
    wd = 1e-3 #weight decay --> note that it might be different on your dataset. See "lr_find" in "demo.ipynb"

    arch = models.resnet34

    data_transformer = (ImageImageList.from_folder(test_path).split_none()
              .label_from_func(lambda x: test_path/x.relative_to(test_path))
              .transform(get_transforms(), size=(820,1024), tfm_y=True)
              .databunch(bs=2).normalize(imagenet_stats, do_y=True))

    #model and hooks
    base_loss = F.l1_loss
    vgg_m = vgg16_bn(True).features.cuda().eval()
    requires_grad(vgg_m, False)
    blocks = [i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)]
    #blocks, [vgg_m[i] for i in blocks]

    #loss
    feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])

    learn = unet_learner(data_transformer, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,
                         blur=True, norm_type=NormType.Weight)

    return learn