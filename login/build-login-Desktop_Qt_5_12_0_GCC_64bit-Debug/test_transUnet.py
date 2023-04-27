# This Python file uses the following encoding: utf-8

# if__name__ == "__main__":
#     pass
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders import custom_transforms as trforms
from dataloaders import tn3k
from dataloaders import utils
from dataloaders.utils import get_dice
from TransUnet.TransUnet import get_transNet
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt as edt
class HausdorffDistance:
    def hd_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # if not np.any(x):
        #     x[0][0] = 1.0
        # elif not np.any(y):
        #     y[0][0] = 1.0

        indexes = np.nonzero(x)
        distances = edt(np.logical_not(y))

        return np.array(np.percentile(distances[indexes], 95))

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (
            pred.shape[1] == 1 and target.shape[1] == 1
            ), "Only binary channel supported"

        pred = (pred > 0.5).byte()
        target = (target > 0.5).byte()
        if torch.sum(pred) == 0:
            pred[0][0][0][0] = 1
            # print(pred)
            # print(torch.sum(pred))
        # print(pred.shape)
        right_hd = torch.from_numpy(
            self.hd_distance(pred.cpu().numpy(), target.cpu().numpy())
            ).float()

        left_hd = torch.from_numpy(
            self.hd_distance(target.cpu().numpy(), pred.cpu().numpy())
            ).float()

        # print(right_hd, ' ', left_hd)

        return torch.max(right_hd, left_hd)

hd_metric = HausdorffDistance()
def fenge():
    print("ceshi")
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #model_name = 'unet'
    load_path = './quanzhong/TransUnet_best.pth'
    input_size = 256
    #test_dataset = 'TN3K'
    net = get_transNet(num_classes=1)
    net.load_state_dict(torch.load(load_path))
    net.cuda()
    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(input_size, input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()])
    test_data = tn3k.TN3K(mode='test', transform=composed_transforms_ts, return_size=True)
    #save_dir = args.save_dir + args.test_fold + '-' + args.test_dataset + os.sep + args.model_name + os.sep
#    save_dir = './results/TN3K/'
    save_dir = '/home/caichengjie/QTproject/1.1/build-1_1-Desktop_Qt_5_14_2_GCC_64bit-Debug/results' + os.sep
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    num_iter_ts = len(testloader)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.cuda()
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        total_jac = 0
        total_dsc = 0
        total_hd_3 = 0
        prec_lists = []
        recall_lists = []

        for sample_batched in tqdm(testloader):
            inputs, labels, label_name, size = sample_batched['image'], sample_batched['label'], sample_batched[
                'label_name'], sample_batched['size']
            inputs = Variable(inputs, requires_grad=False)
            labels = Variable(labels)
            labels = labels.cuda()
            inputs = inputs.cuda()
            outputs = net.forward(inputs)
            prob_pred = torch.sigmoid(outputs)
            jac = utils.get_iou(prob_pred, labels)
            total_jac += jac
            dsc = get_dice(prob_pred, labels)
            total_dsc += dsc
            hd_3 = hd_metric.compute(prob_pred, labels)
            total_hd_3 += hd_3

            prec_list, recall_list = utils.get_prec_recall(prob_pred, labels)
            prec_lists.extend(prec_list)
            recall_lists.extend(recall_list)
            mean_prec = sum(prec_lists) / len(prec_lists)
            mean_recall = sum(recall_lists) / len(recall_lists)

            shape = (size[0, 0], size[0, 1])
            prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear', align_corners=True).cpu().data
            save_data = prob_pred[0]
            save_png = save_data[0].numpy()
            save_png = np.round(save_png)
            save_png = save_png * 255
            save_png = save_png.astype(np.uint8)
            save_path = save_dir + label_name[0]
#            print(save_path)
            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.makedirs(save_path[:save_path.rfind('/')])
            #cv2.imwrite(save_dir + label_name[0], save_png)
            #save_png.save(save_dir + label_name[0])


    print(' iou:' + str(total_jac / len(testloader)))
    print(' dsc:' + str(total_dsc / len(testloader)))
    print(' hd_3:' + str(total_hd_3 / len(testloader)))
    print(' prec:' + str(mean_prec))
    print(' recall:' + str(mean_recall))
    duration = time.time() - start_time
    print("--TN3K contain %d images, cost time: %.4f s, speed: %.4f s." % (
        num_iter_ts, duration, duration / num_iter_ts))
    print("------------------------------------------------------------------")
    jac_out = total_jac / len(testloader)
    dsc_new_out = (2 * jac_out) / (1 + jac_out)
    hd_out = total_hd_3 / len(testloader)
    return [jac_out, dsc_new_out, hd_out]
