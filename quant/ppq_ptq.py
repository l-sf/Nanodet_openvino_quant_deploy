#
# Created by lsf on 2023/6/11.
#

from ppq import *
from ppq.api import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

QUANT_PLATFROM = TargetPlatform.OPENVINO_INT8
MODEL = '../origin_models/nanodet-1.5x-416.onnx'
SAVE_MODEL = 'ppq_int8_models/nanodet-1.5x-416-int8.onnx'
INPUT_SHAPE = [1, 3, 416, 416]
BATCHSIZE = 1
STEPS = 512
CALIBRATION_PATH = 'imgs'
EXECUTING_DEVICE = 'cuda'
REQUIRE_ANALYSE = True
DISPATCH = False
DISPATCH_LAYER_NUM = 5
FINETUNE = True
QS = QuantizationSettingFactory.default_setting()

imgs = []
trans = transforms.Compose([
    transforms.Resize([416, 416]),  # [h,w]
    transforms.ToTensor(),
])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img)
dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)

print('正准备量化你的网络，检查下列设置:')
print(f'TARGET PLATFORM     : {QUANT_PLATFROM.name}')
print(f'NETWORK INPUT SHAPE : {INPUT_SHAPE}')

if FINETUNE:
    QS.lsq_optimization = True  # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.collecting_device = 'cuda'
    QS.lsq_optimization_setting.block_size = 4
    QS.lsq_optimization_setting.lr = 1e-5
    QS.lsq_optimization_setting.gamma = 0
    QS.lsq_optimization_setting.is_scale_trainable = True
    QS.lsq_optimization_setting.steps = 500  # 再训练步数，影响训练时间，500 步大概几分钟

with ENABLE_CUDA_KERNEL():
    qir = quantize_onnx_model(
        onnx_import_file=MODEL,
        calib_dataloader=dataloader,
        calib_steps=STEPS,
        setting=QS,
        input_shape=INPUT_SHAPE,
        collate_fn=lambda x: x.to(EXECUTING_DEVICE),
        platform=QUANT_PLATFROM,
        do_quantize=True
    )

    reports = graphwise_error_analyse(
        graph=qir,
        running_device=EXECUTING_DEVICE,
        steps=STEPS,
        dataloader=dataloader,
        collate_fn=lambda x: x.to(EXECUTING_DEVICE)
    )
    for op, snr in reports.items():
        if snr > 0.1: ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

    if REQUIRE_ANALYSE:
        layerwise_error_analyse(
            graph=qir,
            running_device=EXECUTING_DEVICE,
            interested_outputs=None,
            dataloader=dataloader,
            collate_fn=lambda x: x.to(EXECUTING_DEVICE)
        )
        if DISPATCH:
            # 从大到小排序单层误差
            sensitivity = [(op_name, error) for op_name, error in reports.items()]
            sensitivity = sorted(sensitivity, key=lambda x: x[1], reverse=True)
            print(f'将前 {DISPATCH_LAYER_NUM} 个误差最大的层送上 FP32')
            for op_name, _ in sensitivity[: DISPATCH_LAYER_NUM]:
                QS.dispatching_table.append(operation=op_name, platform=TargetPlatform.FP32)
            print('\n 调度之后的整图累计误差: ')
            reports = graphwise_error_analyse(
                graph=qir,
                running_device=EXECUTING_DEVICE,
                steps=STEPS,
                dataloader=dataloader,
                collate_fn=lambda x: x.to(EXECUTING_DEVICE)
            )

    print(f'网络量化结束，正在生成目标文件: {SAVE_MODEL} \n')
    export_ppq_graph(
        graph=qir, platform=QUANT_PLATFROM,
        graph_save_to=SAVE_MODEL)
