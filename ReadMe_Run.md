# CONVERT PADDLE-OCR MODEL TO PYTORCH MODEL
The PyTorch models are converted from trained models in PaddleOCR.  
### QUICK INSTALLATION 
```bash
pip install -r requirements_converter.txt
```
### Down load PaddleOCR models
[Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/models_list_en.md
)
###  OCR MODELS

```bash
#cls
python3 ./converter/ch_ppocr_mobile_v2.0_cls_converter.py --src_model_path model/ch_ppocr_mobile_v2.0_cls_train

# det 
python ./converter/ch_ppocr_v3_det_converter.py --src_model_path model/en_PP-OCRv3_det_distill_train

# rec 
python converter/convert_rec.py --yaml_path ./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml --src_model_path model/en_PP-OCRv4_rec_train
```

# INFERENCE IN PYTORCH
### QUICK INSTALLATION 
```bash
pip install -r requirements_inference.txt
```

### Inference cli
```commandline
python ./tools/infer/predict_system.py --image_dir ../test.jpeg --det_algorithm DB --det_yaml_path ./configs/det/det_ppocr_v3.yml --det_model_path ./en_ptocr_v3_det_infer.pth --rec_image_shape 3,48,320 --rec_model_path ./en_ptocr_v4_rec_infer.pth --rec_yaml_path ./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml --rec_char_dict_path ./pytorchocr/utils/en_dict.txt
```
### Inference python
```bash
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import cv2
import time
from PIL import Image
from pytorchocr.utils.utility import get_image_file_list, check_and_read_gif
from tools.infer.pytorchocr_utility import draw_ocr_box_txt, parse_args
from tools.infer.predict_system import  TextSystem

if __name__ == '__main__':
    
    args = parse_args()
    args.rec_image_shape =  "3,48,320"
    args.det_yaml_path = "./configs/det/det_ppocr_v3.yml"
    args.rec_yaml_path = "./configs/rec/PP-OCRv4/en_PP-OCRv4_rec.yml"
    args.rec_char_dict_path = "./pytorchocr/utils/en_dict.txt"
    args.use_gpu = True
    # path model converter
    args.rec_model_path = "en_ptocr_v4_rec_infer.pth"
    args.det_model_path = "en_ptocr_v3_det_infer.pth"
    args.cls_model_path = "ch_ptocr_mobile_v2.0_cls_infer.pth"
    ###
    args.image_dir = "test.jpeg"
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score


    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))



```

