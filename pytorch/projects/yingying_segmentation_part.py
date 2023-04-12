import os
import torch
import ocnn
#from tqdm import tqdm
#from config import parse_args
import numpy as np
import torch.nn.functional as F
from easy_mesh_vtk.easy_mesh_vtk import *
#import time
#from multiprocessing import Pool 
import multiprocess as mp
from functools import partial
import shutil
#from pathos.multiprocessing import ProcessingPool as newPool
#from pathos.multiprocessing import ProcessingPool as Pool 
#from cuml.svm import SVC
#from thundersvm import SVC
import argparse

# 通过法向量 对应 八叉树块儿与原始点 的id 
def process(ii, bi, aa, norms):
    idxs = []
    for jj in bi:
        if abs(aa[ii][3]-float(norms[jj][0]))<1e-6 and abs(aa[ii][4]- float(norms[jj][1]))<1e-6 and abs(aa[ii][5]- float(norms[jj][2]))<1e-6 :
            idxs.append([jj,ii])
            bi.remove(jj)
            break
    return idxs
    
def process1(i_node, cell_ids,normals,barycenters):
    idxs = []
    nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
    nei_id = np.where(nei==2)
    for i_nei in nei_id[0][:]:
        if i_node < i_nei:
            cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
            if cos_theta >= 1.0:
                cos_theta = 0.9999
            theta = np.arccos(cos_theta)
            phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
            idxs.append([i_node,i_nei,theta,phi])
    return idxs

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


class NormalizePoints:
  ''' Normalize a point cloud with its bounding sphere

  Args: 
      method: The method used to calculate the bounding sphere, choose from
              'sphere' (bounding sphere) or 'box' (bounding box).
  '''

  def __init__(self, method='sphere'):
    self.method = method

  def __call__(self, points):
    bsphere = ocnn.bounding_sphere(points, self.method)
    radius, center = bsphere[0], bsphere[1:]
    points = ocnn.normalize_points(points, radius, center)
    return points


class TransformPoints:
  ''' Transform a point cloud and the points out of [-1, 1] are dropped. Make
  sure that the input points are in [-1, 1]

  '''

  def __init__(self, distort=False, angle=[0, 180, 0], scale=0.25, jitter=0.25,
               offset=0.0, angle_interval=[1, 1, 1], uniform_scale=False,
               **kwargs):
    self.distort = False
    self.angle = angle
    self.scale = scale
    self.jitter = jitter
    self.offset = 0.016
    self.angle_interval = angle_interval
    self.uniform_scale = uniform_scale

  def __call__(self, points):    
    rnd_angle = [0.0, 0.0, 0.0]
    rnd_scale = [1.0, 1.0, 1.0]
    rnd_jitter = [0.0, 0.0, 0.0]
    if self.distort:
      mul = 3.14159265 / 180.0
      for i in range(3):
        rot_num = self.angle[i] // self.angle_interval[i]
        rnd = np.random.randint(low=-rot_num, high=rot_num+1, dtype=np.int32)
        rnd_angle[i] = rnd * self.angle_interval[i] * mul

      minval, maxval = 1 - self.scale, 1 + self.scale
      rnd_scale = np.random.uniform(low=minval, high=maxval, size=(3)).tolist()
      if self.uniform_scale:  rnd_scale = [rnd_scale[0]]*3

      minval, maxval = -self.jitter, self.jitter
      rnd_jitter = np.random.uniform(low=minval, high=maxval, size=(3)).tolist()

    # The range of points is [-1, 1]
    points = ocnn.transform_points(points, rnd_angle, rnd_scale, rnd_jitter, self.offset)
    return points

def read_ply(file):
   with open(file,'r+') as f:
       a = []
       for i,iline in enumerate(f):
           if i < 13:
               continue
           ilines = iline.strip().split(' ')
           if len(ilines) != 1:
               ilines = [float(ii) for ii in ilines]
               a.append(ilines) 
   return a

def octree_to_ply(new_dir):
    octree_dir = new_dir+r"/octrees/"
    octree_files = os.listdir(octree_dir)
    with open(new_dir+r"/octree2point.txt",'w+') as f:
        for file in octree_files:
            f.write(octree_dir+file+'\n')
    new1 = new_dir+r'/octree2point.txt'
    new2 = new_dir+r'/points'
    cmd_octree2points = 'octree2points --filenames {0} --output_path {1}'.format(new1, new2)
    #cmds = ' '.join(cmd_octree2points)
    #print('\n', cmd_octree2points, '\n')
    os.system(cmd_octree2points)
    
    point_dir = new_dir+r"/points/"    
    point_files = os.listdir(point_dir)
    with open(new_dir+r"/point2ply.txt",'w+') as f1:
        for file in point_files:
            f1.write(point_dir+file+'\n')
    new3 = new_dir+r"/point2ply.txt"
    new4 = new_dir+r"/ply"
    cmd_points2ply = 'points2ply --filenames {0} --output_path {1}'.format(new3, new4)
    #cmds1 = ' '.join(cmd_points2ply)
    #print('\n', cmd_points2ply, '\n')
    os.system(cmd_points2ply)
    #print("do all")
    
def save_ply(filename, points, normals, labels, pts_num=10000):
    data = np.concatenate([points, normals, labels], axis=1)
    header = "ply\nformat ascii 1.0\nelement vertex %d\n" \
      "property float x\nproperty float y\nproperty float z\n" \
      "property float nx\nproperty float ny\nproperty float nz\n" \
      "property float label\nelement face 0\n" \
      "property list uchar int vertex_indices\nend_header\n"
    with open(filename, 'w') as fid:
        fid.write(header % pts_num)
        np.savetxt(fid, data, fmt='%.6f')
        
def ply_to_point(new_dir):
    with open(new_dir+r"/ply2point.txt",'w+') as f:
        f.write(new_dir+r"/original.ply"+'\n')
    new1 = new_dir+r"/ply2point.txt"
    new2 = new_dir
    cmd_ply2points = 'ply2points --filenames {0} --output_path {1} --verbose 0'.format(new1, new2)
    #print('\n', cmd_ply2points, '\n')
    os.system(cmd_ply2points)
    
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_stype', '-m', dest='model_stype', default='latter',type=str,help='choose the model: latter or upper')
    parser.add_argument('--gpu_id','-g', dest='gpu_id',default='0',type=int)
    parser.add_argument('--file','-f', dest='file',default='',type=str)
    parser.add_argument('--input_dir', '-i', dest='input_dir', type=str, default='',
                        help='Directory of input images')
    parser.add_argument('--output_dir', '-o', dest='output_dir', type=str, default='',
                        help='Directory of ouput images')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    gpu_id=args.gpu_id
    model_stype = args.model_stype
    torch.cuda.set_device(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stl_dir = args.input_dir
    save_dir = args.output_dir
    istl = args.file
        
    filenames = istl.strip().split(".")[0]
    if not os.path.exists(r'./tmp'):
        os.mkdir(r'./tmp')
    new_dir = r'./tmp/'+filenames
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    #downsample   
    target_num = 10000
    mesh_original = Easy_Mesh(os.path.join(stl_dir,istl))
    ratio = 1 - target_num/mesh_original.cells.shape[0] # calculate ratio  
    mesh_original.mesh_decimation(ratio)
    mesh_original.get_cell_normals()
    labels = np.zeros([mesh_original.cells.shape[0], 1], dtype=np.int32)
    norms = mesh_original.cell_attributes['Normal']
    cell_centers = (mesh_original.cells[:, 0:3] + mesh_original.cells[:, 3:6] + mesh_original.cells[:, 6:9])/3.0
    pts_num = mesh_original.cells.shape[0]
    #save_ply(r'./tmp/original.ply',cell_centers, norms,labels,pts_num)
    
    #take 1.49s
    save_ply(new_dir+r'/original.ply',cell_centers, norms,labels,pts_num)
    ply_to_point(new_dir)
    

    file2 = new_dir+r'/original.points'   

    sample = np.fromfile(file2, dtype=np.uint8)
    sample = torch.from_numpy(sample)
	# points: 二进制文件 按照dtype=np.float32读取 sample.shape:torch.Size([70015]) == 9999(点云数)*7（坐标xyz\法向量\标签）+22(结构存储大小)
	# points: 二进制文件 按照dtype=np.uint8读取 类型转化为np.uint8（0-255之间） torch.Size([280060])
	# 4个字节(1字节 = 8bit）数：坐标xyz\法向量\标签 ：torch.Size([280060]) points min-max:  tensor(0) tensor(255)   256
    print('sample: ',sample.shape,' ',sample[:20],' ','points min-max: ',torch.min(sample),torch.max(sample),' ',len(torch.unique(sample))) # torch.Size([280060])
    sample = NormalizePoints('sphere')(sample)
   # sample = TransformPoints()(sample)
   # print('sample: ',sample.shape,' ',sample[:20]) # torch.Size([280060])
    octree = Points2Octree()(sample)
    octree_ = ocnn.octree_batch([octree]) 

    octrees = octree_.cuda()
	# torch.Size([5975868]) octree min-max:  tensor(0) tensor(255)   256
    print('octrees: ',octrees,' ',type(octrees),' ',octrees.shape,' ','octree min-max: ',torch.min(octrees),torch.max(octrees),' ',len(torch.unique(octrees)),'\n\n')
  
    octrees_dir = new_dir+r'/octrees'
    if not os.path.exists(octrees_dir):
      os.mkdir(octrees_dir)
    octrees.cpu().numpy().tofile(new_dir+r'/octrees/val.octree')
    octree_to_ply(new_dir)
    
    labels = ocnn.octree_property(octrees, 'label', 8)
    label_mask = labels > -1
    print(torch.where(labels>-1))
	# torch.Size([62968])  tensor(9479, device='cuda:1') 
    print('labels: ',labels.shape,' ',torch.sum(label_mask),'\n\n')
	
    features = ocnn.octree_property(octrees, 'feature', 8)
    # torch.Size([62968])  tensor(9479, device='cuda:1') 
    print('features: ',features.shape,' ','\n\n')
    features = ocnn.octree_feature(octree, 8, False)
    print('features: ',features.shape,' ',features[:,:,[1,2,6,14,20,71848]],'\n')

    #model = ocnn.unet_conv(flags.depth, flags.channel, flags.nout)
    model = ocnn.unet_conv(8, 4, 17)
    #model = ocnn.SegNet_dzh(8, 4, 17)

    #print(model)
    model.cuda()   #take 2 seconds
    #model.to(device, dtype=torch.float)

    # latter
    if model_stype == "latter":
        model_file = r"./models_hehz/model_00300_l.pth"
        if not os.path.exists(model_file):
            model_file = r"./models_hehz/model_00300_l.tar"
    elif model_stype == "upper":
        model_file = r"./models_hehz/model_00300_u.pth"
        if not os.path.exists(model_file):
            model_file = r"./models_hehz/model_00300_u.tar"
    else:
        print("no model")
        sys.exit(0)

    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint)
    del checkpoint
    #model.to(device, dtype=torch.float)

    model.eval()
    #for data in test_loader:
    with torch.no_grad():
        logits = model(octrees)#.to(device, dtype=torch.float)            
        print('logits: ',logits.shape,'\n')  # torch.Size([1, 17, 62968, 1]) 
        prob = F.softmax(logits, dim = 1)  #按照行计算
        prob = prob.squeeze().transpose(0, 1)
    print(prob, prob.shape) #torch.Size([62968, 17])
    probs = prob[label_mask]
    probs = probs.cpu().numpy()
    print(probs, probs.shape) # (9479, 17)


  

