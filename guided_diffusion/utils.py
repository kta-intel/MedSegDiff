
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


softmax_helper = lambda x: F.softmax(x, 1)
sigmoid_helper = lambda x: F.sigmoid(x)


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def maybe_to_torch(d):
    if isinstance(d, list):
        d = [maybe_to_torch(i) if not isinstance(i, torch.Tensor) else i for i in d]
    elif not isinstance(d, torch.Tensor):
        d = torch.from_numpy(d).float()
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):
    if isinstance(data, list):
        data = [i.cuda(gpu_id, non_blocking=non_blocking) for i in data]
    else:
        data = data.cuda(gpu_id, non_blocking=non_blocking)
    return data


class no_op(object):
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

# def staple(a):
#     # a: n,c,h,w detach tensor
#     mvres = mv(a)
#     gap = 0.4
#     if gap > 0.02:
#         for i, s in enumerate(a):
#             r = s * mvres
#             res = r if i == 0 else torch.cat((res,r),0)
#         nres = mv(res)
#         gap = torch.mean(torch.abs(mvres - nres))
#         mvres = nres
#         a = res
#     return mvres

def staple(a, args):
    from .unet_parts import BasicUNet
    model = BasicUNet(n_channels=4, n_classes=1)
    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("Intel(R) Extension for PyTorch* enabled")
        if args.bf16:
            print("BF16 enabled")
            model = ipex.optimize(model, dtype=torch.bfloat16)
            with torch.no_grad(), torch.cpu.amp.autocast():
                return model(a.to(torch.float32))
        else:
            model = ipex.optimize(model)
            return model(a.to(torch.float32))
    else:
        return model(a.to(torch.float32))

def allone(disc,cup):
    disc = np.array(disc) / 255
    cup = np.array(cup) / 255
    res = np.clip(disc * 0.5 + cup,0,1) * 255
    res = 255 - res
    res = Image.fromarray(np.uint8(res))
    return res

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def mv(a):
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def export(tar, img_path=None):
    # image_name = image_name or "image.jpg"
    c = tar.size(1)
    if c == 3:
        vutils.save_image(tar, fp = img_path)
    else:
        s = th.tensor(tar)[:,-1,:,:].unsqueeze(1)
        s = th.cat((s,s,s),1)
        vutils.save_image(s, fp = img_path)

def norm(t):
    m, s, v = torch.mean(t), torch.std(t), torch.var(t)
    return (t - m) / s

def visualize(b, m, args, slice_ID):
    image_0 = b[0, 0].squeeze().cpu().numpy()
    image_1 = b[0, 1].squeeze().cpu().numpy()
    image_2 = b[0, 2].squeeze().cpu().numpy()
    image_3 = b[0, 3].squeeze().cpu().numpy()
    mask = m.squeeze().cpu().numpy()

    import matplotlib.pyplot as plt
    import matplotlib.image
    matplotlib.image.imsave(os.path.join(args.out_dir, "t1_" + str(slice_ID)+ ".jpg"), image_0)
    matplotlib.image.imsave(os.path.join(args.out_dir, "t1ce_" + str(slice_ID)+ ".jpg"), image_1)
    matplotlib.image.imsave(os.path.join(args.out_dir, "t2_" + str(slice_ID)+ ".jpg"), image_2)
    matplotlib.image.imsave(os.path.join(args.out_dir, "flair_" + str(slice_ID)+ ".jpg"), image_3)
    matplotlib.image.imsave(os.path.join(args.out_dir, str(slice_ID)+ ".jpg"), mask)
