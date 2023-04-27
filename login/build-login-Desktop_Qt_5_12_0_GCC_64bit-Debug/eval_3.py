# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from Unet_bot_Transformer.Unet import UNet

def detect_img(img_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("aaa")
    print(img_path)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    data_transform = transforms.Compose([transforms.Resize(size=(256,256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    # img = cv2.imread(source)
    img = Image.open(img_path).convert('RGB')
    size = img.size
#    print(img.size)

    img = data_transform(img)
#    print(img.shape)
    img = torch.unsqueeze(img, dim=0)
    img = img.to(device)
    model = UNet(n_classes=1)

    load_path = './quanzhong/unet_best.pth'
    weight_dict = torch.load(load_path,map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weight_dict)
    print("missing_keys:", missing_keys)
    print("unexpected_keys:", unexpected_keys)
    model = model.to(device)

#    model.eval()
    pred = model(img)
    pred = torch.sigmoid(pred)

    shape = (size[0], size[1])
    prob_pred = F.interpolate(pred, size=shape, mode='bilinear', align_corners=True).cpu().data
    save_data = prob_pred[0]
    save_png = save_data[0].numpy()
    save_png = np.round(save_png)
    save_png = save_png * 255
    save_png = save_png.astype(np.uint8)
    print("zuihou:",save_png.shape)
    save_png = Image.fromarray(save_png)

#    prediction = pred.argmax(1).squeeze(0)
#    prediction = prediction.to("cpu").numpy()
#    save_png = prediction * 255
#    print(save_png[128][128])
#    save_png = save_png.astype(np.uint8)
#    print("zuihou:",save_png.shape)
#    save_png = Image.fromarray(save_png)

#    cv2.imwrite("./result_data/single_result.jpg", save_png)
    save_png.save("./result_data/single_result_unet.jpg")
