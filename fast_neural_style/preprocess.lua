require 'torch'
local M = {}
local function check_input(img)
  assert(img:dim() == 4, 'img must be N x C x H x W')
  assert(img:size(2) == 3, 'img must have three channels') 
end


M.resnet = {}

local resnet_mean = {0.485, 0.456, 0.406}
local resnet_std = {0.229, 0.224, 0.225}

function M.resnet.preprocess(img)
  check_input(img)
  local mean = img.new(resnet_mean):view(1, 3, 1, 1):expandAs(img)
  local std = img.new(resnet_std):view(1, 3, 1, 1):expandAs(img)
  return (img - mean):cdiv(std)
end

function M.resnet.deprocess(img)
  check_input(img)
  local mean = img.new(resnet_mean):view(1, 3, 1, 1):expandAs(img)
  local std = img.new(resnet_std):view(1, 3, 1, 1):expandAs(img)
  return torch.cmul(img, std):add(mean)
end


M.vgg = {}

local vgg_mean = {103.939, 116.779, 123.68}

function M.vgg.preprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return img:index(2, perm):mul(255):add(-1, mean)
end

function M.vgg.deprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(2, perm)
end


return M
