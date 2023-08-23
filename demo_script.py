import argparse
import os
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from guided_diffusion.bratsloader import BRATSDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple, visualize
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms

seed=10
th.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def main():
    args = create_argparser().parse_args()
    
    tran_list = [transforms.Resize((args.image_size,args.image_size)),]
    transform_test = transforms.Compose(tran_list)

    ds = BRATSDataset(args.data_dir,transform_test)
    args.in_ch = 5

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    ######################
    # checkpoint = torch.load(args.model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(new_state_dict)
    ######################
    
    model.eval()
    
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("Intel(R) Extension for PyTorch* enabled")
        if args.bf16:
            print("BF16 enabled")
            model = ipex.optimize(model, dtype=th.bfloat16)
        else:
            model = ipex.optimize(model)

    b, m, slice_ID = next(iter(datal))  #should return an image from the dataloader "data"
    c = th.randn_like(b[:, :1, ...])
    img = th.cat((b, c), dim=1)     #add a noise channel$

    enslist = []

    for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
        model_kwargs = {}

        print("sample " + str(i))
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        
        start_time = time.time()
        
        if args.bf16:
            with th.no_grad(), th.cpu.amp.autocast():
                sample, x_noisy, org, cal, cal_out = sample_fn(
                    model,
                    (args.batch_size, args.in_ch, args.image_size, args.image_size), img,
                    step = args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
        else:
            sample, x_noisy, org, cal, cal_out = sample_fn(
                    model,
                    (args.batch_size, args.in_ch, args.image_size, args.image_size), img,
                    step = args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

        ensemble_step_time = time.time() - start_time
        print("--- %s seconds ---" % (ensemble_step_time))

        co = th.tensor(cal_out)
        if args.version == 'new':
            enslist.append(sample[:,-1,:,:])
        else:
            enslist.append(co)

    ensres = staple(img, args)
    visualize(img, ensres, args, slice_ID)
    # ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
    # vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)
    

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="data/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="model_nonnormalized.pth",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None,
        ipex = False,
        bf16 = False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
