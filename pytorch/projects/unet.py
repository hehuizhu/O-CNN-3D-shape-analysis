import torch
import ocnn
import torch.nn


class UNet(torch.nn.Module):
  def __init__(self, depth, channel_in, nout, nempty=False, interp='linear',
               use_checkpoint=False):
    super(UNet, self).__init__()
    self.depth = depth
    self.channel_in = channel_in
    self.nempty = nempty
    self.use_checkpoint = use_checkpoint
    self.config_network()
    self.stages = len(self.encoder_blocks)

    # encoder
	'''
	conv1:  OctreeConvBnRelu(
	  (conv): OctreeConv(depth=8, channel_in=4, channel_out=32, kernel_size=[3, 3, 3], stride=1, nempty=False)
	  (bn): BatchNorm2d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
	  (relu): ReLU(inplace=True)
	)
	'''
    self.conv1 = ocnn.OctreeConvBnRelu(
        depth, channel_in, self.encoder_channel[0], nempty=nempty)
	print("conv1: ",self.conv1)
    self.downsample = torch.nn.ModuleList(
        [ocnn.OctreeConvBnRelu(depth - i, self.encoder_channel[i],
         self.encoder_channel[i+1], kernel_size=[2], stride=2, nempty=nempty)
         for i in range(self.stages)])
    self.encoder = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(depth - i - 1, self.encoder_channel[i+1],
         self.encoder_channel[i+1], self.encoder_blocks[i], self.bottleneck,
         nempty, self.resblk, self.use_checkpoint) for i in range(self.stages)])

    # decoder
    depth = depth - self.stages
    channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
               for i in range(self.stages)]
    self.upsample = torch.nn.ModuleList(
        [ocnn.OctreeDeConvBnRelu(depth + i, self.decoder_channel[i],
         self.decoder_channel[i+1], kernel_size=[2], stride=2, nempty=nempty)
         for i in range(self.stages)])
    self.decoder = torch.nn.ModuleList(
        [ocnn.OctreeResBlocks(depth + i + 1, channel[i],
         self.decoder_channel[i+1], self.decoder_blocks[i], self.bottleneck,
         nempty, self.resblk, self.use_checkpoint) for i in range(self.stages)])

    # interpolation
    self.octree_interp = ocnn.OctreeInterp(self.depth, interp, nempty)

    # header
    self.header = self.make_predict_module(self.decoder_channel[-1], nout)

  def config_network(self):
    self.encoder_channel = [32, 32, 64, 128, 256]
    self.decoder_channel = [256, 256, 128, 96, 96]
    self.encoder_blocks = [2, 3, 4, 6]
    self.decoder_blocks = [2, 2, 2, 2]
    self.bottleneck = 1
    self.resblk = ocnn.OctreeResBlock2

  def make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
    return torch.nn.Sequential(
        ocnn.OctreeConv1x1BnRelu(channel_in, num_hidden),
        ocnn.OctreeConv1x1(num_hidden, channel_out, use_bias=True))

  def forward(self, octree, pts=None):
    depth = self.depth
    data = ocnn.octree_feature(octree, depth, self.nempty)
    print("data：",data.shape,'\n')  # torch.Size([1, 4, 79808, 1])
    assert data.size(1) == self.channel_in

    # encoder
    convd = [None] * 16
    convd[depth] = self.conv1(data, octree)
    print("convd: ",convd,' ',convd[depth].shape,'\n\n',"depth_i: ")  #torch.Size([1, 32, 79808, 1])

    stages = len(self.encoder_blocks)
    for i in range(stages):
      depth_i = depth - i - 1
      conv = self.downsample[i](convd[depth_i+1], octree)
      convd[depth_i] = self.encoder[i](conv, octree)
      print(depth_i,' ','conv: ',conv.shape,convd[depth_i].shape,'\n',convd[depth_i],'\n\n') 
	  # 7 torch.Size([1, 32, 78552, 1])  torch.Size([1, 32, 78552, 1])  
	  # 6 conv:  torch.Size([1, 64, 69024, 1]) torch.Size([1, 64, 69024, 1])
	  # 5 conv:  torch.Size([1, 128, 29936, 1]) torch.Size([1, 128, 29936, 1])
	  # 4 conv:  torch.Size([1, 256, 4096, 1])  torch.Size([1, 256, 4096, 1])


    # decoder
    depth = depth - stages
    deconv = convd[depth]
    for i in range(stages):
      depth_i = depth + i + 1
      deconv = self.upsample[i](deconv, octree)
      print(depth_i,' ',deconv.shape,' ')
      deconv = torch.cat([convd[depth_i], deconv], dim=1)  # skip connections
      print(deconv.shape,' ','\n\n') 
      deconv = self.decoder[i](deconv, octree)
      print(deconv.shape,' ','\n\n') 
	  # 5 torch.Size([1, 256, 29936, 1])   torch.Size([1, 384, 29936, 1])   torch.Size([1, 256, 29936, 1])
	  # 6 conv:  torch.Size([1, 128, 69024, 1]) torch.Size([1, 192, 69024, 1]) torch.Size([1, 128, 69024, 1])
	  # 7 conv:  torch.Size([1, 96, 78552, 1]) torch.Size([1, 128, 78552, 1]) torch.Size([1, 96, 78552, 1])
	  # 8 conv:  torch.Size([1, 96, 79808, 1])  torch.Size([1, 128, 79808, 1])  torch.Size([1, 96, 79808, 1])

    # point/voxel feature
    feature = deconv
    if pts is not None:
      feature = self.octree_interp(feature, octree, pts)
	  print("feature: ",feature.shape,' ','\n\n') 

    # header
    logits = self.header(feature)
	print("logits: ",logits.shape,' ','\n\n') #torch.Size([1, 17, 79808, 1])
    logits = logits.squeeze().t()  # (1, C, H, 1) -> (H, C)
    return logits

#------------------------------------------------------------------
class Points2Octree:
  ''' Convert a point cloud into an octree
  '''

  def __init__(self, depth=8, full_depth=2, node_dis=False, node_feature=False,
               split_label=False, adaptive=False, adp_depth=4, th_normal=0.1,
               th_distance=1.0, extrapolate=False, save_pts=False, key2xyz=False,
               **kwargs):
    self.depth = depth
    self.full_depth = full_depth
    self.node_dis = True
    self.node_feature = node_feature
    self.split_label = split_label
    self.adaptive = adaptive
    self.adp_depth = adp_depth
    self.th_normal = th_normal
    self.th_distance = th_distance
    self.extrapolate = extrapolate
    self.save_pts = save_pts
    self.key2xyz = key2xyz

  def __call__(self, points):
    octree = ocnn.points2octree(points, self.depth, self.full_depth, self.node_dis,
                                self.node_feature, self.split_label, self.adaptive,
                                self.adp_depth, self.th_normal, self.th_distance,
                                self.extrapolate, self.save_pts, self.key2xyz)
    return octree
	
if __name__=="__main__":
    import numpy as np
    filename="./tmp/40030998_shell_teethup_u/original.points"
    sample = np.fromfile(filename, dtype=np.uint8)
    sample = torch.from_numpy(sample)
    octree = Points2Octree()(sample)
    octree_ = ocnn.octree_batch([octree]) 

    octrees = octree_.cuda()
    # torch.Size([5975868]) octree min-max:  tensor(0) tensor(255)   256
    print('octrees: ',octrees,' ',type(octrees),' ',octrees.shape,' ','octree min-max: ',torch.min(octrees),torch.max(octrees),' ',len(torch.unique(octrees)),'\n\n')
  
    model=UNet(8,4,17)
    model.cuda()
    output=model(octrees)
    print("output: ",output.shape)#torch.Size([79808, 17])
	
	print("----------------------------------")
    depth=8
    channel=4
    channel_out=32
    conv1_test = ocnn.OctreeConv(depth, channel, channel_out, kernel_size=[3,3,3], stride=1,nempty=False)
	data = ocnn.octree_feature(octree, depth, nempty=False)
    print("data：",data.shape,'\n',"weight: ",conv1_test.weights.data.shape,'\n')  # torch.Size([1, 4, 79808, 1])  
    # weight:  torch.Size([32, 108])
	
	
	#  octree_property
	xyz = ocnn.octree_property(octree, 'xyz', depth)
    print('xyz: ',xyz.shape,' ',xyz,'\n\n')
    pts = ocnn.octree_decode_key(xyz)
    print('pts: ',pts.shape,' ',pts,'\n\n')
    xyz_encode = ocnn.octree_encode_key(pts)
    print(xyz_encode.shape,' ',xyz_encode)

    key = ocnn.octree_xyz2key(xyz, depth)
    print('key: ',key.shape,' ',key)

    node_num = ocnn.octree_property(octree, 'node_num', depth)
    print('node_num: ',node_num,' ',node_num.shape)

    node_num_cum = ocnn.octree_property(octree, 'node_num_cum', depth)
    print('node_num_cum: ',node_num_cum,' ',node_num_cum.shape)

    for i in range(1,8):
        child=ocnn.octree_property(octree, 'child', i)
        print('child: ',child,' ',child.shape)

    convd_test=conv1_test(data, octree)
    print(convd_test.shape)