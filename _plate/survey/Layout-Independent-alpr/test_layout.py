import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm

from src.label import Label, dknet_label_conversion
from os.path import splitext, basename
from src.utils import crop_region, image_files_from_folder, nms
from darknet.python.darknet import detect
import darknet.python.darknet as dn


def main(opt):
    vehicle_net  = dn.load_net(bytes(opt.vehicle_netcfg, 'utf-8'), bytes(opt.vehicle_weights, 'utf-8'), 0)
    vehicle_meta = dn.load_meta(bytes(opt.vehicle_dataset, 'utf-8'))

    # Detect vehicle
    imgs_paths = image_files_from_folder(opt.input)
    imgs_paths.sort()
    for ip in tqdm(imgs_paths, ncols=80, desc='Detect vehicle'):
        bname = basename(splitext(ip)[0])
        R, _ = detect(vehicle_net, vehicle_meta, bytes(ip, encoding='utf-8'), thresh=opt.vehicle_threshold)
        if len(R) > 0:
            Iorig = cv2.imread(ip)
            WH = np.array(Iorig.shape[1::-1], dtype=float)
            for i, r in enumerate(R):
                cx, cy, w, h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                tl = np.array([cx - w/2., cy - h/2.])
                br = np.array([cx + w/2., cy + h/2.])
                label = Label(0, tl, br)
                Icar = crop_region(Iorig, label)
                cv2.imwrite('%s/%s_%dcar.jpg' % (opt.output, bname, i), Icar)

    # Detect plate
    lp_net  = dn.load_net(bytes(opt.lp_netcfg, 'utf-8'), bytes(opt.lp_weights, 'utf-8'), 0)
    lp_meta = dn.load_meta(bytes(opt.lp_dataset, 'utf-8'))
    cars_paths = sorted(glob('%s/*car.jpg' % opt.output))
    for ip in tqdm(cars_paths, ncols=80, desc='Detect plate'):
        bname = splitext(basename(ip))[0]
        R, _ = detect(lp_net, lp_meta, bytes(ip, encoding='utf-8'), thresh=opt.lp_threshold)

        if len(R) > 0:
            Ivehicle = cv2.imread(ip)
            WH = np.array(Ivehicle.shape[1::-1], dtype=float)
            Llps = []
            for i, r in enumerate(R):
                cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                tl = np.array([cx - w/2., cy - h/2.])
                br = np.array([cx + w/2., cy + h/2.])
                label = Label(0, tl, br)
                Ilp = crop_region(Ivehicle, label)
                cv2.imwrite('%s/%s_lp.jpg' % (opt.output, bname), Ilp)

    # Rocognize plate
    ocr_net  = dn.load_net(bytes(opt.ocr_netcfg, 'utf-8'), bytes(opt.ocr_weights, 'utf-8'), 0)
    ocr_meta = dn.load_meta(bytes(opt.ocr_dataset, 'utf-8'))
    lps_paths = sorted(glob('%s/*lp.jpg' % opt.output))
    for ip in tqdm(lps_paths, ncols=80, desc='Rocognize plate'):
        bname = splitext(basename(ip))[0]
        R, (width, height) = detect(ocr_net, ocr_meta, bytes(ip, encoding='utf-8'), thresh=opt.ocr_threshold)
        L = dknet_label_conversion(R, width, height)
        L = nms(L, .45)
        L.sort(key=lambda x: x.tl()[0])
        lp_str = ''.join([chr(l.cl()) for l in L])
        print('%s: %s' % (bname, lp_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='test/all')
    parser.add_argument('--output', type=str, default='test/output2')
    parser.add_argument('--vehicle_threshold', type=float, default=0.5)
    parser.add_argument('--vehicle_netcfg', type=str, default='data/Layout-Independent-alpr/vehicle-detection.cfg')
    parser.add_argument('--vehicle_weights', type=str, default='data/Layout-Independent-alpr/vehicle-detection.weights')
    parser.add_argument('--vehicle_dataset', type=str, default='data/Layout-Independent-alpr/vehicle-detection.data')
    parser.add_argument('--lp_threshold', type=float, default=0.01)
    parser.add_argument('--lp_netcfg', type=str, default='data/Layout-Independent-alpr/lp-detection-layout-classification.cfg')
    parser.add_argument('--lp_weights', type=str, default='data/Layout-Independent-alpr/lp-detection-layout-classification.weights')
    parser.add_argument('--lp_dataset', type=str, default='data/Layout-Independent-alpr/lp-detection-layout-classification.data')
    parser.add_argument('--ocr_threshold', type=float, default=0.5)
    parser.add_argument('--ocr_netcfg', type=str, default='data/Layout-Independent-alpr/lp-recognition.cfg')
    parser.add_argument('--ocr_weights', type=str, default='data/Layout-Independent-alpr/lp-recognition.weights')
    parser.add_argument('--ocr_dataset', type=str, default='data/Layout-Independent-alpr/lp-recognition.data')
    opt = parser.parse_args()

    main(opt)