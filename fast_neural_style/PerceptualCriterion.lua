require 'torch'
require 'nn'
require 'fast_neural_style.ContentLoss'
require 'fast_neural_style.StyleLoss'
require 'fast_neural_style.DeepDreamLoss'

local layer_utils = require 'fast_neural_style.layer_utils'
local crit, parent = torch.class('nn.PerceptualCriterion', 'nn.Criterion')

function crit:__init(args)
  args.content_layers = args.content_layers or {}
  args.style_layers = args.style_layers or {}
  args.deepdream_layers = args.deepdream_layers or {}
  
  self.net = args.cnn
  self.net:evaluate()
  self.content_loss_layers = {}
  self.style_loss_layers = {}
  self.deepdream_loss_layers = {}

  for i, layer_string in ipairs(args.content_layers) do
    local weight = args.content_weights[i]
    local content_loss_layer = nn.ContentLoss(weight, args.loss_type)
    layer_utils.insert_after(self.net, layer_string, content_loss_layer)
    table.insert(self.content_loss_layers, content_loss_layer)
  end


  for i, layer_string in ipairs(args.deepdream_layers) do
    local weight = args.deepdream_weights[i]
    local deepdream_loss_layer = nn.DeepDreamLoss(weight)
    layer_utils.insert_after(self.net, layer_string, deepdream_loss_layer)
    table.insert(self.deepdream_loss_layers, deepdream_loss_layer)
  end
  
  layer_utils.trim_network(self.net)
  self.grad_net_output = torch.Tensor()

end


function crit:setStyleTarget(target)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('none')
  end
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('capture')
  end
  self.net:forward(target)
end

function crit:setContentTarget(target)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('none')
  end
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('capture')
  end
  self.net:forward(target)
end


function crit:setStyleWeight(weight)
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer.strength = weight
  end
end


function crit:setContentWeight(weight)
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer.strength = weight
  end
end


function crit:updateOutput(input, target)
    if target then
    self:setContentTarget(target)
  end
  if target.style_target then
    self.setStyleTarget(target.style_target)
  end

  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    content_loss_layer:setMode('loss')
  end
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    style_loss_layer:setMode('loss')
  end

  local output = self.net:forward(input)

  self.grad_net_output:resizeAs(output):zero()

  self.total_content_loss = 0
  self.content_losses = {}
  self.total_style_loss = 0
  self.style_losses = {}
  for i, content_loss_layer in ipairs(self.content_loss_layers) do
    self.total_content_loss = self.total_content_loss + content_loss_layer.loss
    table.insert(self.content_losses, content_loss_layer.loss)
  end
  for i, style_loss_layer in ipairs(self.style_loss_layers) do
    self.total_style_loss = self.total_style_loss + style_loss_layer.loss
    table.insert(self.style_losses, style_loss_layer.loss)
  end
  
  self.output = self.total_style_loss + self.total_content_loss
  return self.output
end


function crit:updateGradInput(input, target)
  self.gradInput = self.net:updateGradInput(input, self.grad_net_output)
  return self.gradInput
end

