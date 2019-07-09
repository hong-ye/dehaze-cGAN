require 'nngraph'

function defineGour_net(input_nc, output_nc, ngf)
	local netG = nil
   	 -- input is (nc) x 256 x 256
   	 local e0 = - nn.SpatialConvolution(input_nc, ngf, 5, 5, 1, 1, 2, 2)
	 local e0_1 = e0   - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf , 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf) 

   	 local e1 = e0_1   - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
   	 -- input is (ngf) x 128 x 128

   	 local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

   	 -- input is (ngf * 2) x 64 x 64

   	 local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)

   	 -- input is (ngf * 4) x 32 x 32

   	 local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 16 x 16

   	 local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 8 x 8

   	 local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 4 x 4

   	 local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 2 x 2

   	 local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 1 x 1



   	 local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)

   	 -- input is (ngf * 8) x 2 x 2

   	 local d1 = {d1_,e7} - nn.JoinTable(2)

   	 local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)

   	 -- input is (ngf * 8) x 4 x 4

   	 local d2 = {d2_,e6} - nn.JoinTable(2)

   	 local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)

   	 -- input is (ngf * 8) x 8 x 8

   	 local d3 = {d3_,e5} - nn.JoinTable(2)

   	 local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 *2 , ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 16 x 16

   	 local d4 = {d4_,e4} - nn.JoinTable(2)

   	 local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8*2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)

   	 -- input is (ngf * 4) x 32 x 32

   	 local d5 = {d5_,e3} - nn.JoinTable(2)

   	 local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4*2 , ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

   	 -- input is (ngf * 2) x 64 x 64

   	 local d6 = {d6_,e2} - nn.JoinTable(2)

   	 local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2*2 , ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)

   	 -- input is (ngf) x128 x 128

   	 local d7 = {d7_,e1} - nn.JoinTable(2)

   	 local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*2 , ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)

   	 -- input is (nc) x 256 x 256
   	-- local d8 = {d8_,e0_2} - nn.JoinTable(2)
   	 local d9 = d8 - nn.ReLU(true)  - nn.SpatialConvolution(ngf , output_nc, 3, 3, 1, 1, 1, 1)
	-- local d9_ = d9 - nn.Identity()
   	 local o1 = d9 - nn.Tanh()
   	 netG = nn.gModule({e0},{o1})
	return netG
end


function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
    
        local netD = nn.Sequential()
        
        netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 3, 3, 1, 1, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        
        nf_mult = 1
        for n = 1, n_layers-1 do 
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 3, 3, 1, 1, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        end
        
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 3, 3, 1, 1, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 3, 3, 3, 1, 1, 1, 1))       
        netD:add(nn.Sigmoid())
        
        return netD
    end
