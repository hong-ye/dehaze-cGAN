require 'torch'
require 'nn'

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')


function ContentLoss:__init(strength, loss_type)
  parent.__init(self)
  self.strength = strength or 1.0
  self.loss = 0
  self.target = torch.Tensor()

  self.mode = 'none'
  loss_type = loss_type or 'L1'
  if loss_type == 'L2' then
    self.crit = nn.MSECriterion()
  elseif loss_type == 'L1' then
    self.crit = nn.AbsCriterion()
  else
    error(string.format('Invalid loss_type "%s"', loss_type))
  end
end


function ContentLoss:updateOutput(input)
  if self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  elseif self.mode == 'loss' then
    self.loss = self.strength * self.crit:forward(input, self.target)
  end
  self.output = input
  return self.output
end


function ContentLoss:updateGradInput(input, gradOutput)
  if self.mode == 'capture' or self.mode == 'none' then
    self.gradInput = gradOutput
  elseif self.mode == 'loss' then
    self.gradInput = self.crit:backward(input, self.target)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  end
  return self.gradInput
end


function ContentLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
