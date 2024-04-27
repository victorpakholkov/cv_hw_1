import os
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from model import Yolo
from utils import nms, test_and_draw_mAP

model_weight_path = ''
categories = []
test_index = ''
image_path = ''
results_dir = ''

def test(net, batch_size):
  
	lines = os.listdir(image_path)

	with torch.no_grad():
		idx = 0
		while idx < len(lines):
			to_idx = min(len(lines), idx + batch_size)
			batch_lines = lines[idx:to_idx]
			idx = to_idx

			X = torch.Tensor([])
			X = X.to('cuda')

			S = torch.Tensor([])
			S = S.to('cuda')

			for line in batch_lines:
				image_name = line
				this_path = os.path.join(image_path, image_name)

				with Image.open(this_path) as img:
					s = torch.Tensor([float(img.size[0]), float(img.size[1])])
					s = s.to('cuda')
					S = torch.cat((S, s.unsqueeze_(0)), 0)

					rimg = torchvision.transforms.functional.resize(img, (448, 448))
					t = torchvision.transforms.ToTensor()(rimg).to('cuda')
					X = torch.cat((X, t.unsqueeze_(0)), 0)

			YHat = net(X)
			for i, yhat in enumerate(YHat):
        			# nms
				yhat = nms(yhat)

				yhat = yhat.reshape((-1, 30))
				W = S[i][0]
				H = S[i][1]

				category_detected = [False for _ in range(20)]

				for j in range(yhat.shape[0]):
					x, y, w, h, iou = yhat[j][0:5]

					# calculate cell index
					xidx = j % 7
					yidx = j // 7

					# transform cell relative coordinates to image relative coordinates
					x = (x + xidx) / 7.0
					y = (y + yidx) / 7.0

					score, cat = yhat[j][10:30].max(dim=0)
					if iou * score < 0.1: continue

					file_name = f'comp_det_test_{categories[cat]}.txt'
					file_path = os.path.join(results_dir, file_name)

					category_detected[cat] = True

					with open(file_path, 'a+', encoding='utf-8', newline='\n') as f:
						x1 = max(1, int((x - w / 2) * W))
						y1 = max(1, int((y - h / 2) * H))
						x2 = min(int(W), int((x + w / 2) * W))
						y2 = min(int(H), int((y + h / 2) * H))
						conf = round(float(score * iou), 6)
						f.write(f'{batch_lines[i]} {conf} {x1}.000000 {y1}.000000 {x2}.000000 {y2}.000000\n')



if __name__ == '__main__':
  
  resnet18 = torchvision.models.resnet18(pretrained=True)
  backbone = nn.Sequential(*list(resnet18.children())[:-2])
  net = Yolo(backbone, backbone_out_channels=512)

  net.to('cuda')
  net.load_state_dict(torch.load(model_weight_path))
  net.eval()
  
  test(net, 64)
  test_and_draw_mAP(net, test_iter_raw, 'cuda')

  
  

