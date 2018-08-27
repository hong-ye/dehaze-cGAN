require 'torch'
require 'nn'

local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Criterion')

function TVLoss:__init(input)
  parent.__init(self)
--  self.strength = 1e-3
  self.x_diff = torch.Tensor():typeAs(input)
  self.y_diff = torch.Tensor():typeAs(input)
  self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):zero()
end

function TVLoss:forward(input)
  self.output = 0
  return self.output
end
-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:backward(input)
  local B,C, H, W = input:size(1), input:size(2), input:size(3), input:size(4)
  self.x_diff:resize(B, C, H - 1, W - 1)
  self.y_diff:resize(B, C, H - 1, W - 1)
  self.x_diff:copy(input[{{},{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{},{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{},{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{},{}, {2, -1}, {1, -2}}])
  self.gradInput[{{},{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{},{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{},{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
 -- self.gradInput:mul(self.strength)
 -- self.gradInput:add(gradOutput)
  return self.gradInput
end
