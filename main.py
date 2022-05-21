
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import segmentation_models_pytorch as smp ## version 0.2.1 #pip install segmentation-models-pytorch
import torch  # pip install torch torchvision torchaudio
from torch.autograd import Variable
import onnx # pip install -U pip && pip install onnx-simplifier
from onnxsim import simplify
import onnxruntime
import numpy as np
import albumentations as albu # pip install albumentations

from matplotlib import pyplot as plt # pip install matplotlib
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
import cv2


ENCODER = "resnet18"#"se_resnet50"#
ENCODER_WEIGHTS = None
ACTIVATION = 'sigmoid'  # Was None
device = torch.device('cpu')
model = smp.Unet(
    encoder_name=ENCODER,
    # encoder_weights=ENCODER_WEIGHTS,
    classes=1,
    activation=ACTIVATION,
    decoder_attention_type='scse' # should be disabled if not 0.2.1
).to(device)


# For 300x300 frames
model.eval()
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 480
X = torch.ones((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
torch_out = model(X)

onnx_model_path = "./onnxmodel.onnx"

# export the model
input_names = [ "input" ]
output_names = [ "output" ]

print('exporting model to ONNX...')
torch.onnx.export(model,
                  X,
                  onnx_model_path,
                  export_params=True,
                  verbose=True,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=input_names,
                  output_names=output_names, # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}},
                  opset_version=12)

# torch.onnx.export(
#     model,
#     X, # Dummy input for shape
#     onnx_model_path,
#     opset_version=12,
#     do_constant_folding=True,
# )


onnx_simple_path = "./simpleonnxmodel.onnx"
onnx_model = onnx.load(onnx_model_path)
print("result:", onnx.checker.check_model(onnx_model))
model_simpified, check = simplify(onnx_model)
onnx.save(model_simpified, onnx_simple_path)



ort_session = onnxruntime.InferenceSession(onnx_model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
def num_numpy(tensor):
    return np.expand_dims(tensor, 0)

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(X)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
# import blobconverter # pip install blobconverter
# #
# blobconverter.from_onnx(
#     model=onnx_simple_path,
#     output_dir="./model.blob",
#     data_type="FP16",
#     shaves=6,
#     use_cache=False,
#     optimizer_params=[]
# )
image_raw = cv2.imread("./1581997697_baby-penguin.jpg", cv2.IMREAD_COLOR)

height = 320
width = 480
h, w = image_raw.shape[:2]

aug = albu.Compose([
    albu.Resize(height, width),
    albu.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)), ])
result = aug(
    image=image_raw
)

image = result['image']
image = (image - np.min(image)) / (np.max(image) - np.min(image))
# image = np.expand_dims(image,0)
# img_trans = image.transpose((2, 1, 0))
result = ToTensor()(image=image)['image']

# torch_out_2 = model(result)
result = result.expand(1, 3, 320, 480)
ort_inputs_new = {ort_session.get_inputs()[0].name: to_numpy(result)}
ort_outs_new = ort_session.run(None, ort_inputs)
img_val = Variable(result).to(device)
all_outputs = model(img_val)

outputs = np.transpose(all_outputs.cpu().detach().numpy()[0], (1, 2, 0))
thresh = 0.95
outputs_1 = np.zeros_like(outputs)
outputs_1[outputs < thresh] = 0
outputs_1[outputs >= thresh] = 1
outputs = outputs_1
import cv2
prediction_mask = outputs
prediction_mask = cv2.resize(prediction_mask, (w, h))

prediction_mask[prediction_mask < 0.5] = 0
prediction_mask[prediction_mask >= 0.5] = 1
plt.subplot(1,2,1)
plt.imshow(outputs)
plt.subplot(1,2,2)
plt.imshow(ort_outs_new[0][0,0,:,:])
plt.show()

