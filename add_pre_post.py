#
# Created by lsf on 2023/5/11.
#

from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Core, Layout, Type, serialize


model_path = "quant/ppq_int8_models/nanodet-1.5x-416-int8.xml"


# Step1: 创建OpenVINO运行时
core = Core()

# Step2: 读取模型、加载模型
model = core.read_model(model_path)

# Step3: 通过OpenVINO预处理器将预处理功能集成到模型中

ppp = PrePostProcessor(model)
ppp.input().tensor() \
    .set_element_type(Type.u8) \
    .set_color_format(ColorFormat.BGR) \
    .set_layout(Layout('NHWC'))
ppp.input().model().set_layout(Layout('NCHW'))
ppp.output().tensor().set_element_type(Type.f32)
ppp.input().preprocess() \
    .convert_element_type(Type.f32) \
    .convert_color(ColorFormat.RGB) \
    .mean([123.675, 116.28, 103.53]) \
    .scale([58.395, 57.12, 57.375])
model = ppp.build()


# Step4: 使用预处理保存模型
serialize(model, 'nanodet-full.xml', 'nanodet-full.bin')

