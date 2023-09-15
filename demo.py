import os
import time

import torch
import torchvision.transforms as transforms

from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.utils import post_process, visualize, dice_coeff
from guided_diffusion.script_util import create_model_and_diffusion

import gradio as gr

def segment_tumor(input_img, optimization):
    diffusion_steps = 20

    tran_list = [transforms.Resize((256,256))]
    transform_test = transforms.Compose(tran_list)

    ds = BRATSDataset(f'data/MICCAI_BraTS2020/{input_img}', transform_test)

    datal = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True)

    model, diffusion = create_model_and_diffusion(image_size=256, in_ch=5, diffusion_steps=diffusion_steps, version=1)
    model.eval()

    if optimization=='Intel(R) Extension for PyTorch*':
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.bfloat16)

    b, m, _ = next(iter(datal)) 
    c = torch.randn_like(b[:, :1, ...])
    img = torch.cat((b, c), dim=1) 
    
    model_kwargs = {}

    sample_fn = diffusion.p_sample_loop_known
    start_time = time.time()

    if optimization=='Intel(R) Extension for PyTorch*' or optimization=='Stock PyTorch* (BF16)':
        with torch.no_grad(), torch.cpu.amp.autocast():
            sample, _, _, _, _ = sample_fn(
                model,
                (1, 5, 256, 256), img, optimization,
                step = diffusion_steps,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
    else:
        with torch.no_grad():
            sample, _, _, _, _ = sample_fn(
                model,
                (1, 5, 256, 256), img, 
                optimization,
                step = diffusion_steps,
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )

    # if optimization=='Intel(R) Extension for PyTorch*':
    #     with torch.no_grad(), torch.cpu.amp.autocast():
    #         sample, _, _, _, _ = sample_fn(
    #             model,
    #             (1, 5, 256, 256), img, optimization,
    #             step = diffusion_steps,
    #             clip_denoised=True,
    #             model_kwargs=model_kwargs,
    #         )
    # else:
    #     with torch.no_grad():
    #         sample, _, _, _, _ = sample_fn(
    #             model,
    #             (1, 5, 256, 256), img, 
    #             optimization,
    #             step = diffusion_steps,
    #             clip_denoised=True,
    #             model_kwargs=model_kwargs,
    #         )

    ensemble_step_time = f"{time.time() - start_time:.2f}"
    post_processed_m = post_process(sample)

    t1, t1ce, t2, flair = visualize(b, post_processed_m)
        
    output_img = [(t1,'T1') , (t1ce,'T1CE'), (t2,'T2'), (flair,'FLAIR')]
        
    dice_score = f"Dice Coefficient: {dice_coeff(post_processed_m, m):.2f}"

    return output_img, dice_score, ensemble_step_time


# demo = gr.Interface(
#     fn=segment_tumor, 
#     inputs=[gr.Textbox(label="Input Image"), 
#             gr.Dropdown(['Disabled', 'Enabled'], label="Optimization")], 
#     outputs=[gr.Image(label="Tumor Segmentation Prediction"),
#              gr.Textbox(label="Accuracy"),
#              gr.Textbox(label="Inference time (s)")], 
#     live=False, 
#     title="ACCELERATE DIFFUSION MODELS FOR MEDICAL SEGMENTATION",
#     allow_flagging='never',
#     css="footer {visibility: hidden}",
#     theme='Taithrah/Minimal',
#     article='Test'
# )


with gr.Blocks(theme='Taithrah/Minimal') as demo:
    gr.Markdown("""
    # <p style="text-align: center;"> ACCELERATE DIFFUSION MODELS FOR MEDICAL IMAGE SEGMENTATION </p>
    ## <p style="text-align: center;"> USING INTEL® AI OPTIMIZATIONS </p>
    """)
    with gr.Row():
        with gr.Column():
            input_img = gr.Dropdown(['IM_001', 
                                     'IM_002',
                                     'IM_003',
                                     'IM_004',
                                     'IM_005',
                                     'IM_006',
                                     'IM_007',
                                     'IM_008',
                                     'IM_009',
                                     'IM_010'], label="Input Image")
            optimization = gr.Dropdown(['Stock PyTorch*',
                                        # 'Stock PyTorch* (BF16)', 
                                        'Intel(R) Extension for PyTorch*'], label="Optimization")
            segment_btn = gr.Button(value="Segment")

            gr.Markdown("""
                        | Intel(R) Extension for PyTorch* | Intel(R) Advanced Matrix Extensions |
                        | :-----------------------------: | :---------------------------------: |
                        | <center>![](file/IPEX.png)</center>  | <center>![](file/amx.png)</center> |
                        """)
            
        with gr.Column():
            output_img = gallery = gr.Gallery(label="Tumor Segmentation Prediction", 
                                              show_download_button=False,
                                              elem_id="gallery", 
                                              columns=[4], rows=[1], 
                                              height=256)
            dice_score = gr.Textbox(label="Accuracy")
            ensemble_step_time = gr.Textbox(label="Inference time (s)")
            
    with gr.Accordion("Additional details", open=False):
        gr.Markdown("""
        The model used in this demonstration is a replication of the cited work and not the original model. Accuracy and performance does not reflect the published material. Results may vary.  
        <br/>
        Hardware Configuration: 4th Gen Intel(R) Xeon(R) Scalable: 1-node, 1x Intel(R) Xeon(R) Platinum 8488C. Ubuntu 22.04.2 LTS, Linux Kernel 6.2.0-1011-aws. AWS EC2 m7i.xlarge.  
        <br/>
        Software Configuration: PyTorch 2.0.1+cpu, Intel(R) Extension for PyTorch* 2.0.100+cpu   
        <br/>
        **MedSegDiff**  
        J. Wu, H. Fang, Y. Zhang, Y. Yang, and Y. Xu, *MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model*, Proceedings of Machine Learning Research – nnn:1–17, 2023 [[Code](https://github.com/WuJunde/MedSegDiff), [Paper](https://openreview.net/pdf?id=Jdw-cm2jG9)]  
        <br/>
        **BraTS 2020 Dataset**  
        B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694 (opens in a new window)  
        S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117(opens in a new window)  
        S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)  
        S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q(opens in a new window)  
        S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al.,(opens in a new window) "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF

        """) 

    gr.Markdown("""
### Notices & Disclaimers:  

This tool is intended for demonstration purposes only and does not provide medical advice. It is not a substitute for professional medical diagnosis, advice, or treatment. Do not disregard professional medical advice in seeking treatment because of something you see in this demo or from the underlying implementation. If you think you are experiencing a medical event, immediately call your doctor or emergency services.  

This tool is not intended for use which could create obligations under HIPAA, Intel is not a Covered Entity under HIPAA and Intel makes no representations that the demonstration of some capabilities of this tool satisfy HIPAA requirements. If you are (or becomes a Covered Entity or Business Associate, as defined in HIPAA), you cannot use this tool for any purpose or in any manner involving Protected Health Information (as defined in HIPAA).  

Performance varies by use, configuration and other factors. Learn more at www.Intel.com/PerformanceIndex. Performance results are based on testing as of dates shown in configurations and may not reflect all publicly available updates. See backup for configuration details. No product or component can be absolutely secure. Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.
                """)

    segment_btn.click(fn=segment_tumor, 
                      inputs=[input_img, 
                              optimization], 
                      outputs=[output_img,
                               dice_score,
                               ensemble_step_time]
                     )
  
    

demo.launch(share=True)