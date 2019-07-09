%PSNR - Peak signal-to-noise ratio and gradient w.r.t. each pixel.

%   Author: Uwe Schmidt, TU Darmstadt (uwe.schmidt@gris.tu-darmstadt.de)
%
%   This file is part of the implementation as described in the CVPR 2014 paper:
%   Uwe Schmidt and Stefan Roth. Shrinkage Fields for Effective Image Restoration.
%   Please see the file LICENSE.txt for the license governing this code.

function [f,g] = psnr_web(x, y)

  MAX = 255;
  npixels = numel(x);
  delta = x(:) - y(:);
  mse = mean(delta.^2);
  
  f = 20*log10(MAX) - 10*log10(mse);
  g = - ( 20 / (log(10)*mse*npixels) ) * delta;
end
