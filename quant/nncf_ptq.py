#
# Created by lsf on 2023/5/11.
#

import nncf
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from openvino.runtime import Core, serialize

MODEL_PATH = '../workspace/origin/nanodet-1.5x-320.xml'
INT8_PATH = "../workspace/NNCF_INT8_openvino_model/nanodet-1.5x-320-int8.xml"
CALIBRATION_PATH = 'imgs'
BATCHSIZE = 1
SUBSET_SIZE = 512

imgs = []
trans = transforms.Compose([
    transforms.Resize([320, 320]),  # [h,w]
    # transforms.ToTensor()
])
for file in os.listdir(path=CALIBRATION_PATH):
    path = os.path.join(CALIBRATION_PATH, file)
    img = Image.open(path).convert('RGB')
    img = trans(img)
    imgs.append(img)

dataloader = DataLoader(dataset=imgs, batch_size=BATCHSIZE)


# def transform_fn(data_item):
#     return {'image': data_item.numpy()}


nncf_calibration_dataset = nncf.Dataset(data_source=dataloader)


preset = nncf.QuantizationPreset.MIXED

model = core = Core()
ov_model = core.read_model(MODEL_PATH)
quantized_model = nncf.quantize(
    ov_model, nncf_calibration_dataset, preset=preset, subset_size=SUBSET_SIZE
)

serialize(quantized_model, INT8_PATH)
