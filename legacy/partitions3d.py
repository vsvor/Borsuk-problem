import torch 
import numpy as np
import json
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from itertools import combinations



pinf = 1E6

no_impr_steps = 100
min_lr = 1e-8
comp_heu = 1000
penalty_coef = 5.0

def lineeq(n,c,x):
    return n.dot(x)-c

class CPoly: 
             
    def __init__(self, center=[0.5,0.5], pts = []):
        self.lines = []
        self.center = center
        self.points = pts.copy()
        #pts.append(pts[0])
        for i in range(len(pts)-1):
            self.addline(pts[i],pts[i+1])
        self.addline(pts[0],pts[len(pts)-1])


    def addline(self,a,b):
        d = np.array(b) - np.array(a)
        n = np.array([-d[1],d[0]])
        n = n/np.linalg.norm(n)
        c = n.dot(a)
        if n.dot(self.center)-c < 0:
            n = -n
            c = -c
        self.lines.append([n,c])

    def mindist(self,p):
        m = pinf
        for l in self.lines:
            v = l[0].dot(p)-l[1]
            if v<m:
                m = v
        return m

    def prepare(self):
        n = len(self.lines)
        self.U = np.zeros([n,2])
        self.v = np.zeros(n)
        k = 0
        for l in self.lines:
            self.U[k,:] = l[0]
            self.v[k] = l[1]
            k += 1


class CPolytop:     
                
    def __init__(self, d, center=[], pts = []):
        self.planes = []
        if len(center)>0:
            self.center = center
        else:
            self.center = np.zeros(d)
        self.points = np.array(pts.copy())

        self.d = d
        #pts.append(pts[0])
        if len(pts)>0:
          hull = ConvexHull(self.points)
          for eq in hull.equations:
            e1 = eq[:-1]
            e2 = eq[-1]
            if e1.dot(self.center)+e2<0:
              e1 = -1.0*e1
              e2 = -1.0*e2
            self.planes.append([e1,e2])


    def mindist(self,p):
        m = pinf
        for l in self.lines:
            v = l[0].dot(p)+l[1]
            if v<m:
                m = v
        return m

    def prepare(self):
        nl = len(self.planes)
        self.U = np.zeros([nl,self.d])
        self.v = np.zeros(nl)
        k = 0
        for l in self.planes:
            self.U[k,:] = l[0]
            self.v[k] = l[1]
            k += 1
'''
class CPolytopSph(CPolytop):
    def __init__(self, d, center=[], pts = []):


    def mindist(self,p):

    def prepare(self):
'''

eps = 1E-6
eps2 = eps**2

def eq_pts(a,b):
  c = np.array(a)-np.array(b)
  return c.dot(c) < eps2

def pt_index(pts,a):
  k = 0
  for p in pts:
    if eq_pts(p,a):
      return k
    k += 1
  return -1

from itertools import product

def extra_points(d):
  pts = np.array(list(product([-1,1], repeat=d))) * 100.0
  return pts


def filter_eqs(A):
    res = []
    for i in range(A.shape[0]):
        if pt_index(res,A[i,:])==-1:
            res.append(A[i,:])
    return np.array(res)

class OptDiagramNd: 

  def pt_index(self,p):
    for i in range(len(self.points)):
      if eq_pts(p, self.points[i]):
        return i
    return -1

  def set_mask(self):
    n = len(self.points)
    self.mask = np.zeros((n,n),dtype=bool)
    for p in self.polygons:
      for i,j in combinations(p,2):
        self.mask[i,j] = True
    self.mask =  torch.BoolTensor(self.mask)

  def __init__(self,  bound, points = None, saved = None, penalty_coef = 5.0):
    self.bound = bound
    self.d = bound.d
    self.bound_vert_n = len(bound.points)
    self.penalty_coef = penalty_coef
    if saved is None:
#                self.bound = bound
#                self.d = bound.d
                m0 = len(points)
                ext_pts = np.array(list(points) + list(extra_points(self.d)))
                vor = Voronoi(ext_pts)
                self.vor = vor

                finite_regions = []
                finite_reg_points = []

                k = 0
                for reg in vor.regions:
                  if len(reg)>0 and not (-1 in reg):
                    finite_regions.append(reg)
                    cur = -1
                    for i in range(len(vor.point_region)):
                      if vor.point_region[i] == k:
                        cur = i
                        break
                    finite_reg_points.append(cur)
                  k += 1
                self.points = []
                for p in bound.points:
                  self.points.append(p)
                self.bound_vert_n = len(self.points)

                box = ConvexHull(bound.points)

                self.polygons = []

                k = 0
                for reg in finite_regions:
                  #print(reg)
                  regp = vor.vertices[reg]
                  hull = ConvexHull(regp)
                  all_eq = filter_eqs(np.concatenate((box.equations,hull.equations)))


                  pt = vor.points[finite_reg_points[k]]
                  hs = HalfspaceIntersection(all_eq,pt)

                  cur = []
                  for p in hs.intersections:
                    i =  pt_index(self.points,p)
                    if i>=0:
                      cur.append(i)
                    else:
                      self.points.append(p)
                      cur.append(len(self.points)-1)
                  if len(cur)>0:
                    self.polygons.append(cur)
                  k += 1
    else:
                self.points = []
                for p in saved[1]:
                    self.points.append(p)
                self.polygons = saved[0].copy()


    self.points = np.array(self.points)
    dist_lines = bound.U.dot(np.transpose(self.points)) + bound.v.reshape((len(bound.planes),1))
    self.line_link = [] 
    m = len(bound.planes)
    n = len(self.points)
    self.n = n

    self.set_mask()

    self.mask_lines = np.zeros((m,n - self.bound_vert_n),dtype=bool)
    for i in range(dist_lines.shape[0]):
      for j in range(self.bound_vert_n, dist_lines.shape[1]):
        if abs(dist_lines[i,j])<eps:
          self.line_link.append([j,i])
          self.mask_lines[i,j - self.bound_vert_n] = True

    self.mask_lines =  torch.BoolTensor(self.mask_lines)
    self.bound_pt_tensor = torch.Tensor(bound.points)
    self.line_U = torch.tensor(bound.U)
    self.line_v = torch.tensor(bound.v.reshape((len(bound.planes),1)))

  def forward(self,x):
    x_all = torch.cat((self.bound_pt_tensor,x))
    n = len(x_all)
    x2 = torch.square(x_all)
    x2s = torch.sum(x2,1)
    distm =  - 2* x_all.mm(x_all.t()) + x2s + x2s.reshape((n,1))
    diags = torch.sqrt(torch.masked_select(distm,self.mask))
    dist_lines = torch.abs(torch.masked_select(self.line_U.mm(x.t()) + self.line_v, self.mask_lines)) 
    self.err = torch.max(dist_lines)
    return torch.max(diags) + penalty_coef*self.err    

  def diams(self, tol = 0.99):     
    x_all = torch.tensor(self.points)
    n = len(x_all)
    x2 = torch.square(x_all)
    x2s = torch.sum(x2,1)
    distm =  - 2* x_all.mm(x_all.t()) + x2s + x2s.reshape((n,1))
    res = []
    diags = torch.sqrt(torch.masked_select(distm,self.mask))
    m = torch.max(diags)
    m2 = m*m
    for i in range(n):
      for j in range(n):
        if self.mask[i,j]:
          if distm[i,j] > m2*tol:
            res.append([i,j])
    return m, res

  def proj(self,x): 
    for p,l in self.line_link:
      p1 = p - self.bound_vert_n
      d = self.bound.lines[l][0].dot(x[p1]) - self.bound.lines[l][1]
      res = x[p1] - d*self.bound.lines[l][0]
      x[p1,:] = res

  def draw_poly(self, x = None, diams = None):
    plt.figure(figsize=(12,12))
    if x is None:
      points = self.points
    else:
      points = np.concatenate((self.bound_pt_tensor.numpy(),x))
    for p in self.polygons:
      cur = []
      for v in p:
        cur.append(points[v])
      plt.fill(*zip(*cur), alpha=0.4)
    #print(points[0])

    if not diams is None:
      for d in diams:
        p1 = points[d[0]]
        p2 = points[d[1]]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],linestyle='dotted')

    plt.axis('equal')
    plt.xlim(min(points[:,0])-0.1, max(points[:,0]) + 0.1)
    plt.ylim(min(points[:,1])-0.1, max(points[:,1]) + 0.1)

messages_iter = 1000
almost_inf = 100.0

from torch.autograd import Variable

class OptPartition(): 

  def __init__(self, points, n, n_iter_circ = 2000, n_iter_part = 70000, lr_start = 0.0005, lr_decay = 0.97, prec_opt = 1/1000, diam_tol = 0.99, messages = 1):
    self.n_iter1 = n_iter_circ   
    self.n_iter2 = n_iter_part   
    self.n = n
    self.d = len(points[0])      
    self.poly = CPolytop(self.d, center = np.zeros(self.d), pts = points)
    self.poly.prepare()
    self.U = torch.Tensor(self.poly.U)
    self.v = torch.Tensor(self.poly.v).reshape((len(self.poly.planes),1))
    self.lr_start = lr_start    
    self.lr_decay = lr_decay    
    self.messages = messages    
    self.tol = diam_tol         
    self.prec_opt = prec_opt    

  def random_packing(self):    
    U = torch.Tensor(self.poly.U)
    v =  torch.Tensor(self.poly.v).reshape((len(self.poly.planes),1))
    x = Variable(torch.rand((self.n,self.d)) - 0.5, requires_grad = True)
    lr = 0.003
    optimizer = torch.optim.Adam([x], lr=lr)
    mask = torch.BoolTensor([[i<j for i in range(self.n)] for j in range(self.n)])
    for i in range(self.n_iter1):
      optimizer.zero_grad()
      x2 = torch.square(x)
      x2s = torch.sum(x2,1)
      distm =  - 2* x.mm(x.t()) + x2s + x2s.reshape((self.n,1))
      dist_points = 0.5*torch.sqrt(torch.masked_select(distm,mask))
      dist_lines =   U.mm(x.t()) + v
      y = -torch.min(torch.min(dist_points),torch.min(dist_lines))
      y.backward(retain_graph=True)
      optimizer.step()

      if (i + 1) % messages_iter == 0:
        #lr = lr*0.8
        for g in optimizer.param_groups:
          g['lr'] = lr

        if self.messages>=2:
          print(i+1,' min_L = ', torch.min(dist_points), '  min_P = ', torch.min(dist_lines),'   r = ', abs(float(y.detach().numpy())))



    return x.detach().numpy()

  def random_packing_sph(self):
    #U = torch.Tensor(self.poly.U)
    #v =  torch.Tensor(self.poly.v).reshape((len(self.poly.planes),1))
    x = Variable(torch.rand((self.n,self.d)) - 0.5, requires_grad = True)
    lr = 0.003
    optimizer = torch.optim.Adam([x], lr=lr)
    mask = torch.BoolTensor([[i<j for i in range(self.n)] for j in range(self.n)])
    for i in range(self.n_iter1):
      optimizer.zero_grad()
      x2 = torch.square(x)
      x2s = torch.sum(x2,1)
      distm =  - 2* x.mm(x.t()) + x2s + x2s.reshape((self.n,1))
      dist_inv = 1.0/torch.sqrt(torch.masked_select(distm,mask))
      #dist_lines =   U.mm(x.t()) + v
      sq_dist_sph = torch.square(x2s - 1.0)
      #y = -torch.min(torch.min(dist_points),torch.min(dist_lines))
      y = torch.sum(dist_inv) + 10.0*torch.sum(sq_dist_sph)
      y.backward(retain_graph=True)
      optimizer.step()

      if (i + 1) % messages_iter == 0:
        #lr = lr*0.8
        for g in optimizer.param_groups:
          g['lr'] = lr

        if self.messages>=2:
          print(i+1,' min_L = ', torch.min(dist_points), '  min_P = ', torch.min(dist_lines),'   r = ', abs(float(y.detach().numpy())))
    #print(y.detach().numpy())
    return x.detach().numpy()*0.3


  def random_partition(self,x0):      
    #self.od = OptDiagramNd(self.poly,x0)
    try:
        self.od = OptDiagramNd(self.poly,x0)
    except:
        return almost_inf, None

    t = np.array(self.od.points[self.od.bound_vert_n:])
    x = torch.tensor(t, requires_grad = True)
    lr = self.lr_start
    optimizer = torch.optim.Adam([x], lr=lr)
    yp = almost_inf
    y_best = almost_inf
    no_impr = 0
    i = 0
    precise = False
    while True:
      #print(i)
      optimizer.zero_grad()
      y = self.od.forward(x)
      y.backward(retain_graph=True)
      optimizer.step()
      ycur = y.detach().numpy()
      if ycur < y_best:
        y_best = ycur
        no_impr = 0
      else:
        no_impr +=1
      if (i + 1) % messages_iter == 0:
        if self.messages>=2:
          print(i+1, '   d = ',float(ycur))
        if (i>self.n_iter2):
          break

        #if (y.detach().numpy()>yp):
        #  if precise:
        #    break
        #  else:
        #    lr = lr*self.prec_opt  
                                    
        #    precise = True
        yp = y.detach().numpy()
        #######################3
      if no_impr > no_impr_steps:
        lr = lr * self.lr_decay
        if lr < min_lr:
            break
        if (ycur - self.best_d)/lr > comp_heu:
            break
        for g in optimizer.param_groups:
          g['lr'] = lr
      i+=1
    self.od.points[self.od.bound_vert_n:] = x[:,:].detach().numpy()
    d, dlist = self.od.diams(tol = self.tol)
    if self.messages>=3:
      self.od.draw_poly(diams=dlist)
      plt.show()
    return d.detach().numpy(), self.od.points

  def multiple_runs(self,m):  
    best_d = almost_inf
    self.best_d = almost_inf
    for i in range(m):

          x = self.random_packing_sph()
          if np.isnan(x).any():
            if self.messages>=1:
              print('NaN in data')
            continue
          d, p = self.random_partition(x)
          new_rec = ''
          if d<best_d:
            best_d = d
            self.best_d = d
            self.best_p = p.copy()
            self.best_poly = self.od.polygons.copy()
            self.best_err = self.od.err
            new_rec = '*'
          if self.messages>=1:
            print('run {0}/{1}, d_max = {2}, err = {3} {4}'.format(i+1,m,d,self.od.err,new_rec))

    #if self.messages>=1:
    self.od.points = self.best_p.copy()
    self.od.polygons = self.best_poly
    self.od.set_mask()
    d, dlist = self.od.diams(tol = self.tol)
      #self.od.draw_poly(diams=dlist)
      #plt.show()
    #self.best_d = best_d
    return float(best_d)

  def from_file(self,filename):
    with open(filename,'r') as f:
        l = f.readlines()
        poly = json.loads(l[3])
        points = json.loads(l[4])
        self.od = OptDiagramNd(self.poly, saved = (poly,points))
        self.od.set_mask()
        d, dlist = self.od.diams(tol = self.tol)
        self.best_d = d.detach().numpy()
        return self.best_d

  def to_file(self,filename):  
    with open(filename,'w') as f:
      f.write(str(self.n)+'\n')
      f.write(str(self.best_d)+'\n')
      f.write(str(len(self.od.points))+'\n') 
      json.dump(self.od.polygons,f) 
      f.write('\n')
      json.dump(self.od.points.tolist(),f) 

  def to_string(self):
    res = ""
    res += (str(self.n)+'\n')
    res += (str(self.best_d)+'\n')
    res += (str(len(self.od.points))+'\n')
    res += json.dumps(self.od.polygons) 
    res += '\n'
    res +=json.dumps(self.od.points.tolist()) 
    return res

  def diams(self, tol = 0.01):
    l = self.best_d
    res = []
    for p in self.od.polygons:
        for i,j in combinations(p,2):
            l1 = np.linalg.norm(self.od.points[i]-self.od.points[j])
            if 1.0-tol <l1/l<1.0+tol:
                res.append((i,j))
    return res
