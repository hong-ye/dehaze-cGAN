require 'nngraph'

function defineGour_net(input_nc, output_nc, ngf)
	 local netG = nil
   	 -- input is (nc) x 256 x 256
   	 local e0 = - nn.SpatialConvolution(input_nc, ngf, 5, 5, 1, 1, 2, 2)
	 local e0_1 = e0   - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf , 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf) 
		
   	 local e1 = e0_1 - nn.LeakyReLU(0.2, true)  - nn.SpatialConvolution(ngf, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)

   	 -- input is (ngf) x 128 x 128

   	 local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

   	 -- input is (ngf * 2) x 64 x 64
   	 local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)

   	 -- input is (ngf * 4) x 32 x 32

   	 local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 16 x 16

   	 local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 16)


   	 local d4_ = e5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16 , ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

   	 -- input is (ngf * 8) x 16 x 16

   	 local d4 = {d4_,e4} -  nn.JoinTable(2)

   	 local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8*2 , ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)

   	 -- input is (ngf * 4) x 32 x 32

   	 local d5 = {d5_,e3} -  nn.JoinTable(2)

   	 local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4*2 , ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)

   	 -- input is (ngf * 2) x 64 x 64

   	 local d6 = {d6_,e2} - nn.JoinTable(2)

   	 local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2*2 , ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)

   	 -- input is (ngf) x128 x 128

   	 local d7 = {d7_,e1} -  nn.JoinTable(2)

   	 local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*2 , ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)

   	 -- input is (nc) x 256 x 256
   	 local d9 = d8  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d10 = d9 - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d10_ = {d10,d8} - nn.CAddTable()
   	 local d11 = d10_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d12 = d11  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d12_ = {d12,d10_} - nn.CAddTable()
   	 local d13 = d12_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d14 = d13  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d14_ = {d14,d12_} - nn.CAddTable()
   	 local d15 = d14_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d16 = d15  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d16_ = {d16,d14_} - nn.CAddTable()
   	 local d17 = d16_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d18 = d17  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d18_ = {d18,d16_} - nn.CAddTable()
   	 local d19 = d18_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d20 = d19  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d20_ = {d20,d18_} - nn.CAddTable()
   	 local d21 = d20_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d22 = d21  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d22_ = {d22,d20_} - nn.CAddTable()
   	 local d23 = d22_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d24 = d23  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d24_ = {d24,d22_} - nn.CAddTable()
   	 local d25 = d24_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d26 = d25  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d26_ = {d26,d24_} - nn.CAddTable()
   	 local d27 = d26_  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
   	 local d28 = d27  - nn.SpatialConvolution(ngf , ngf, 3, 3, 1, 1, 1, 1)
	 local d28_ = {d28,d26_} - nn.CAddTable()
   	 local d29 = d28_  - nn.SpatialConvolution(ngf , output_nc, 3, 3, 1, 1, 1, 1)

   	 netG = nn.gModule({e0},{d29})
	return netG
end


function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
    
        local netD = nn.Sequential()       
        netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 3, 3, 1, 1, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf))
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
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 3, 3, 1, 1, 1, 1))       
        netD:add(nn.Sigmoid())
        
        return netD
    end
