#position-invariant transform

import math
import os
import sys
import time
import numpy as np
from PIL import Image
import torch
from itertools import product


class PIT_module:
	def __init__(self, w, h, fovx = 0, fovy = 0, isPITedSize = False):
		'''
		w and h: the width and height of input image
		fovx, fovy: intrinsic parameter of camera. One is enough, the other would be calculated through the aspect ratio. Provided in Radian system.
		isPITedSize = False: the size(w and h) comes from an original image.
		isPITedSize = Ture: the size(w and h) comes from a PITed image. (Used when need to reverse PIT a PITed image, and don't know the original size)

		If you need to transform a image circularly (original->PITed->original), don't need to create two PIT_module.
		You can do this by setting the "reverse" parameter in the "pit" function.
		'''
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.plain_width, self.plain_height, self.arc_width, self.arc_height = 0, 0, 0, 0

		if isPITedSize:
			self.arc_width, self.arc_height = w, h
			self.cal_fov_with_PITed_image_size(fovx, fovy)
			self.cal_plain_size()
		else:
			self.plain_width, self.plain_height = w, h
			self.aspect_ratio = self.plain_width / self.plain_height
			self.cal_fov_with_original_image_size(fovx, fovy)
			self.cal_arc_size()

		self.arc_pos_list = 0
		self.plain_pos_list = 0

	def cal_fov_with_PITed_image_size(self, fovx, fovy):
		'''Calculate the focal length (and fovx or fovy if not provided)'''
		self.fovx = fovx
		self.fovy = fovy
		if not fovx == 0 and not fovy == 0:
			self.fx = self.arc_width / fovx
			self.fy = self.arc_height / fovy
		elif not fovx == 0:
			self.fx = self.arc_width / fovx
			self.fy = self.fx
			self.fovy = self.arc_height / self.fy
		elif not fovy == 0:
			self.fy = self.arc_height / fovy
			self.fx = self.fy
			self.fovx = self.arc_width/ self.fx

	def cal_fov_with_original_image_size(self, fovx, fovy):
		self.fovx = fovx
		self.fovy = fovy
		if not fovx == 0 and not fovy == 0:
			pass
		elif not fovx == 0:
			self.fovy = 2 * math.atan(1 / self.aspect_ratio * math.tan(fovx / 2))
		else: # not fovy == 0:
			self.fovx = 2 * math.atan(self.aspect_ratio * math.tan(fovy / 2))

		self.fx = self.plain_width / (2 * math.tan(self.fovx / 2)) #focal length x
		self.fy = self.plain_height / (2 * math.tan(self.fovy / 2)) #focal length y

	def cal_arc_size(self):
		'''known the size of original image, calculate the size of PITed image'''
		self.arc_width = int(2 * math.atan(self.plain_width / self.fx / 2) * self.fx)
		self.arc_height = int(2 * math.atan(self.plain_height / self.fy / 2) * self.fy)

	def cal_plain_size(self):
		'''known the size of PITed image, calculate the size of original image'''
		self.plain_width = int(2 * math.tan(self.arc_width / self.fx / 2) * self.fx)
		self.plain_height = int(2 * math.tan(self.arc_height / self.fy / 2) * self.fy)

	def coord_plain_to_arc(self, pos_list):
		x = pos_list[:,0] + 0.5
		y = pos_list[:,1] + 0.5
		u = self.fx * (self.fovx / 2 - torch.atan((self.plain_width / 2 - x) / self.fx))
		v = self.fy * (self.fovy / 2 - torch.atan((self.plain_height / 2 - y) / self.fy))
		u -= 0.5
		v -= 0.5
		new_pos_list = torch.stack([u, v], dim=1)
		return new_pos_list

	def coord_arc_to_plain(self, pos_list):
		u = pos_list[:,0] + 0.5
		v = pos_list[:,1] + 0.5
		x = self.plain_width / 2 - self.fx * torch.tan(self.fovx / 2 - u / self.fx)
		y = self.plain_height / 2 - self.fy * torch.tan(self.fovy / 2 - v / self.fy)
		x -= 0.5
		y -= 0.5
		new_pos_list = torch.stack([x,y], dim = 1)
		return new_pos_list

	def coord_plain_to_arc_scalar(self, x, y):
		'''used for pit annotations'''
		x += 0.5
		y += 0.5
		u = self.fx * (self.fovx / 2 - math.atan((self.plain_width / 2 - x) / self.fx))
		v = self.fy * (self.fovy / 2 - math.atan((self.plain_height / 2 - y) / self.fy))
		u -= 0.5
		v -= 0.5
		return u,v

	def coord_arc_to_plain_scalar(self, u, v):
		'''used for pit annotations'''
		u += 0.5
		v += 0.5
		x = self.plain_width / 2 - self.fx * math.tan(self.fovx / 2 - u / self.fx)
		y = self.plain_height / 2 - self.fy * math.tan(self.fovy / 2 - v / self.fy)
		x -= 0.5
		y -= 0.5
		return x,y

	def create_pos_list(self, w, h):
		a = list(range(w))  # x
		b = list(range(h))  # y
		pos = [i for i in product(a, b)]
		pos = torch.Tensor(pos).to(self.device)
		return pos

	def limit_range(self, t, min_value, max_value):
		'''set the value in t less than min_value to min_value, more than max_value to max_value'''
		t[t < min_value] = min_value
		t[t > max_value] = max_value
		return t

	def pit_cal_pos_list(self):
		'''used for pit'''
		self.arc_pos_list = self.create_pos_list(self.arc_width, self.arc_height)
		self.pos_in_plain = self.coord_arc_to_plain(self.arc_pos_list)
		self.pos_in_plain_nearest = self.change_2d_pos_into_1d_index(self.limit_range(torch.round(self.pos_in_plain), 0, self.plain_width - 1), self.plain_width)
		self.pos_in_plain_4_vtx = self.cal_4_vertex(self.pos_in_plain, self.plain_width, self.plain_height)
		self.pos_in_plain_16_vtx = self.cal_16_vertex(self.pos_in_plain, self.plain_width, self.plain_height)

	def rpit_cal_pos_list(self):
		'''used for reversed pit'''
		self.plain_pos_list = self.create_pos_list(self.plain_width, self.plain_height)
		self.pos_in_arc = self.coord_plain_to_arc(self.plain_pos_list)
		self.pos_in_arc_nearest = self.change_2d_pos_into_1d_index(self.limit_range(torch.round(self.pos_in_arc), 0, self.arc_width - 1), self.arc_width)
		self.pos_in_arc_4_vtx = self.cal_4_vertex(self.pos_in_arc, self.arc_width, self.arc_height)
		self.pos_in_arc_16_vtx = self.cal_16_vertex(self.pos_in_arc, self.arc_width, self.arc_height)

	def change_2d_pos_into_1d_index(self, pos, w):
		'''the function torch.take() needs 1d index'''
		x = pos[:, 0]
		y = pos[:, 1]
		pos_1dim = (y * w + x).long()
		return pos_1dim

	def cal_4_vertex(self, pos_in_float, w, h):
		'''used for bilinear interpolation'''
		x = self.limit_range(pos_in_float[:, 0], 0, w - 1)  #
		y = self.limit_range(pos_in_float[:, 1], 0, h - 1)

		pos_4_vtx = [[0 for i in range(2)] for i in range(2)]

		x_list = [torch.floor(x), torch.ceil(x)]  # x_list
		y_list = [torch.floor(y), torch.ceil(y)]  # y_list

		for i in range(2):
			for j in range(2):
				pos = torch.stack([x_list[i], y_list[j]], dim=1)
				pos_4_vtx[i][j] = self.change_2d_pos_into_1d_index(pos, w)

		return pos_4_vtx

	def cal_16_vertex(self, pos_in_float, w, h):
		'''used for bicubic interpolation'''
		x = self.limit_range(pos_in_float[:, 0], 0, w - 1)  #
		y = self.limit_range(pos_in_float[:, 1], 0, h - 1)
		x_list = [self.limit_range(torch.floor(x), 0, w - 1), torch.floor(x), torch.ceil(x),
		          self.limit_range((torch.ceil(x) + 1), 0, w - 1)]
		y_list = [self.limit_range(torch.floor(y), 0, h - 1), torch.floor(y), torch.ceil(y),
		          self.limit_range((torch.ceil(y) + 1), 0, h - 1)]

		pos_16_vtx = [[0 for i in range(4)] for i in range(4)]
		for i in range(4):
			for j in range(4):
				pos = torch.stack([x_list[i], y_list[j]], dim=1)
				pos_16_vtx[i][j] = self.change_2d_pos_into_1d_index(pos, w)
		return pos_16_vtx

	def nearest_interpolation(self, this_channel, pos):
		res = torch.take(this_channel, pos)
		return res

	def bilinear_interpolation(self, this_channel, real_pos, pos_4_vtx):
		pix_lt = torch.take(this_channel, pos_4_vtx[0][0])  # left top
		pix_lb = torch.take(this_channel, pos_4_vtx[0][1])  # left bottom
		pix_rt = torch.take(this_channel, pos_4_vtx[1][0])  # right top
		pix_rb = torch.take(this_channel, pos_4_vtx[1][1])  # right bottom

		x = real_pos[:, 0]
		y = real_pos[:, 1]
		xfrac, yfrac = torch.frac(x), torch.frac(y)

		t = pix_lt + (pix_rt - pix_lt) * xfrac
		b = pix_lb + (pix_rb - pix_lb) * xfrac
		res = t + (b - t) * yfrac
		return res

	def W(self, x):
		'''used for bicubic interpolation'''
		x = torch.abs(x)
		a = 1 - 2 * x.pow(2) + x.pow(3)
		b = 4 - 8 * x + 5 * x.pow(2) - x.pow(3)
		c = torch.zeros_like(x)
		res1 = torch.where(x <= 1, a, c)
		res2 = torch.where(x < 2, b, c)
		res = torch.where(res1 == c, res2, res1)
		return res

	def bicubic_interpolation(self, this_channel, real_pos, pos_16_vtx):
		'''
		The caculation method comes from:
			https://blog.csdn.net/yycocl/article/details/102588362
		'''
		x = real_pos[:, 0]
		y = real_pos[:, 1]
		u,v = torch.frac(x), torch.frac(y) #u,v

		pix = [[0 for i in range(4)] for i in range(4)]
		for i in range(4):
			for j in range(4):
				pix[i][j] = torch.take(this_channel, pos_16_vtx[i][j])

		res = 0

		for i in range(4):
			for j in range(4):
				res += pix[i][j] * self.W((i - 1 - u)) * self.W((j - 1 - v))
		res = self.limit_range(res, 0, 255)
		return res

	def pit(self, im, interpolation = 'bilinear', reverse = False, ori_w = 0, ori_h = 0):
		'''
		im: image in torch tensor format. [N,C,H,W]
		reverse: False = PIT, True = rPIT
		interpolation: 'nearest' or 'bilinear' or 'bicubic'
		ori_w/ori_h: the width and height of original image.
			if reverse == True, the better to provide them to avoid error.
		'''
		pix = im  #n,c,h,w  torch.Size([2, 19, 180, 360])
		#print(im.shape)
		start_dim = len(pix.shape)
		assert start_dim == 3 or start_dim == 4
		if start_dim == 3: #n,h,w
			pix = pix[:,None,:,:]

		self.n, self.c = pix.shape[0], pix.shape[1]

		pos = 0
		pos_4_vtx = 0
		pos_nearest = 0
		new_w, new_h = 0, 0

		if not reverse:
			new_w, new_h = self.arc_width, self.arc_height
			if type(self.arc_pos_list) == type(0):
				self.pit_cal_pos_list()
			pos = self.pos_in_plain
			pos_nearest = self.pos_in_plain_nearest
			pos_4_vtx = self.pos_in_plain_4_vtx
			pos_16_vtx = self.pos_in_plain_16_vtx
		if reverse:
			if ori_w:
				self.plain_width = ori_w
			if ori_h:
				self.plain_height = ori_h
			new_w, new_h = self.plain_width, self.plain_height
			if type(self.plain_pos_list) == type(0):
				self.rpit_cal_pos_list()
			pos = self.pos_in_arc
			pos_nearest = self.pos_in_arc_nearest
			pos_4_vtx = self.pos_in_arc_4_vtx
			pos_16_vtx = self.pos_in_arc_16_vtx

		batch_new = []
		for i in range(self.n):
			pix_new = []
			for j in range(self.c):
				res = 0
				this_channel = pix[i,j,:,:].float().squeeze()  # dtype:torch.float32
				if interpolation == 'nearest' or interpolation == 1:
					res = self.nearest_interpolation(this_channel, pos_nearest)
				elif interpolation == 'bilinear' or interpolation == 2:
					res = self.bilinear_interpolation(this_channel, pos, pos_4_vtx)
				elif interpolation == 'bicubic' or interpolation == 3:
					res = self.bicubic_interpolation(this_channel, pos, pos_16_vtx)
				else:
					print('"' + interpolation + '" is not a interpolation mode!')

				res = torch.round(res).reshape(new_w, new_h).int()  # [h,w]
				res = torch.transpose(res, 0, 1)  # [h,w]
				pix_new.append(res)
			pix_new = torch.stack(pix_new, dim=0)
			batch_new.append(pix_new)
		batch_new = torch.stack(batch_new, dim = 0)
		if start_dim == 3:
			batch_new = batch_new.flatten(0,1)
		return batch_new

'''convert tensor and image'''
def tensor_to_image(t):
	if t.shape[1] == 1: #gray image
		im = t[0, 0, ...]
	else:    #rgb image
		im = t[0,...].permute(1,2,0)  # n c h w -> h w c
	if not t.device == 'cpu':
		im = im.cpu()
	im = im.numpy().astype('uint8')
	im = Image.fromarray(im)
	return im

def image_to_tensor(im):
	t = torch.from_numpy(np.array(im)) #[h,w,channel]
	if len(t.shape) == 2:  #gray images
		t = t[None, None, ...]  # n c h w
	else:   #RGB images
		t = t.permute(2, 0, 1)[None, ...]  #n c h w
	return t


if __name__ == "__main__":
	'''Usage of PIT_module'''

	interpolation = 2

	'''testing for gray images (1 channel) '''
	im_path = 'test_images/gray.png'
	im = Image.open(im_path)
	t = image_to_tensor(im).cuda()  #create input

	width, height = t.shape[3], t.shape[2]
	proj = PIT_module(width, height, fovx=math.pi / 2)

	t_new = proj.pit(t, interpolation = interpolation)
	im_new = tensor_to_image(t_new)
	im_new.save('test_images/gray_pit_tensor.png')

	t_cycle = proj.pit(t_new, interpolation = interpolation, reverse = True, ori_w = width, ori_h = height)
	im_cycle = tensor_to_image(t_cycle)
	im_cycle.save('test_images/gray_cycle_tensor.png')

	'''testing for RGB images (3 channel)'''
	im_path = 'test_images/RGB.png'
	im = Image.open(im_path)
	t = image_to_tensor(im).cuda()  #create input

	width, height = t.shape[3], t.shape[2]
	proj = PIT_module(width, height, fovx=math.pi / 2)

	t_new = proj.pit(t, interpolation = interpolation)
	im_new = tensor_to_image(t_new)
	im_new.save('test_images/RGB_pit_tensor.png')

	t_cycle = proj.pit(t_new, interpolation = interpolation, reverse = True, ori_w = width, ori_h = height)
	im_cycle = tensor_to_image(t_cycle)
	im_cycle.save('test_images/RGB_cycle_tensor.png')
