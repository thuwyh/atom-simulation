# 参数
from __future__ import annotations
from copy import deepcopy
import json
from typing import List, Optional, Tuple, Union
import math
from pydantic import BaseModel
from enum import Enum
from tqdm import tqdm
import sys
from argparse import ArgumentParser
import yaml
from cachetools import cached, LRUCache
sys.setrecursionlimit(10000)  # Setting the limit to 1500

class Direction(Enum):
    POS_X=1
    NEG_X=-1
    POS_Y=2
    NEG_Y=-2
    POS_Z=3
    NEG_Z=-3
    OTHER = 0

class Plane(Enum):
    Z_IS_0 = 1
    Y_IS_0 = 2
    Z_IS_C = 3
    Y_IS_B = 4
    X_IS_0 = 5
    X_IS_A = 6
    NOT_BOUNDARY = 7

class CollisionState(Enum):
    INVALID = 0 # 会碰到处PT外的其他结构原子
    VALID = 1 # 可行点，没碰到任何结构原子
    SEMI_VALID_PT = 2 # 除了pt没碰到其他任何结构原子，但距离不超过阈值
    VALID_PT = 3 # 除了pt没碰到其他任何结构原子
    
    

class MeshPoint(BaseModel):
    x: int
    y: int
    z: int
    collision_state: Optional[CollisionState]=None
    # real_x: Optional[float]=None
    # real_y: Optional[float]=None
    # real_z: Optional[float]=None
    father:MeshPoint=None
    valid_end_point:bool=None
    semi_valid_end_point:bool=None
    # def model_post_init(self, __context):
    #     self.real_x = self.x/mesh_num
    #     self.real_y = self.y/mesh_num
    #     self.real_z = self.z/mesh_num


    def is_coord_valid(self, a, b, c):
        if self.x<0 or self.x>a:
            return False
        if self.y<0 or self.y>b:
            return False
        if self.z<0 or self.z>c:
            return False
        return True
    
    def is_collision_state_valid(self):
        # assert self.collision_state is not None
            # self.collision_state = collision_detection(self, start_p, main_dir, mesh)
        if self.collision_state!=CollisionState.INVALID:
            return True
        return False
    
    def is_valid_end_point(self, start_plane, mesh:Mesh):
        if self.valid_end_point is not None:
            return self.valid_end_point
        plane, _ = get_start_plane_and_main_direction(self, mesh)
        if plane!=Plane.NOT_BOUNDARY and \
            plane!=start_plane and \
            self.collision_state==CollisionState.VALID_PT:
            self.valid_end_point = True
        else:
            self.valid_end_point = False
        return self.valid_end_point
    
    def is_semi_valid_end_point(self, start_plane, mesh:Mesh):
        if self.semi_valid_end_point is not None:
            return self.semi_valid_end_point
        plane, _ = get_start_plane_and_main_direction(self, mesh)
        if plane!=Plane.NOT_BOUNDARY and \
            plane!=start_plane and \
            self.collision_state==CollisionState.SEMI_VALID_PT:
            self.semi_valid_end_point = True
        else:
            self.semi_valid_end_point = False
        return self.semi_valid_end_point
    
    def get_str_rep(self):
        return f"{self.x}_{self.y}_{self.z}"
    
    def __hash__(self):
        return hash((self.x, self.y, self.z))

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

class Mesh:

    def __init__(self, param_path:str, structure_atoms:List[List]):
        config = load_config(param_path)
        self.atom_radius = config['atom_radius']
        self.a = config['a']
        self.b = config['b']
        self.c = config['c']
        self.mesh_num = config['mesh_num']
        self.probe_r = config['probe_r']
        self.mesh_radius = math.sqrt(3)/self.mesh_num # 真实坐标半径
        self.probe_mesh_delta = config['probe_mesh_delta'] # 网格坐标下在主方向正交平面内横向探索的delta
        # 先分解成单个方向上的范围，方便后面是用。这样处理后正交平面探索范围是个方形
        self.single_direction_delta = list(range(-self.probe_mesh_delta, self.probe_mesh_delta+1)) 
        
        self.structure_atoms = structure_atoms
        self.max_plane_distance = config['max_plane_distance']
        # 把归一化坐标转化成真实坐标
        for idx in range(len(self.structure_atoms)):
            self.structure_atoms[idx][0]*=self.a
            self.structure_atoms[idx][1]*=self.b
            self.structure_atoms[idx][2]*=self.c

        self.visited_point = {}
        self.all_points = {}
        self.pbar = tqdm(total=int(self.a*self.b*self.c*self.mesh_num*self.mesh_num*self.mesh_num),
                         desc="visited points")

    def visit(self, p:MeshPoint):
        self.visited_point[p.get_str_rep()]=p.collision_state
    
    def has_visited_before(self, p:MeshPoint):
        point_id = p.get_str_rep()
        return point_id in self.visited_point and p.collision_state.value<=self.visited_point[point_id].value
    
    def get_mesh_point(self, x, y, z):
        str_rep = f"{x}_{y}_{z}"
        if str_rep in self.all_points:
            return self.all_points[str_rep]
        new_mesh_point = MeshPoint(
            x=x,
            y=y,
            z=z
        )
        self.all_points[str_rep] = new_mesh_point
        self.pbar.update(1)
        return new_mesh_point
    
    def shutdown(self):
        self.pbar.close()

class Point(BaseModel):
    x: float
    y: float
    z: float

# 归一化坐标x*a是真实坐标（原子半径坐标）
# 归一化坐标x*a*mesh_num是网格坐标
# 以下是超参数


# 读数据

# dedupe mesh points
def dedupe_mesh_points(points:List[MeshPoint]):
    seen = set()
    ret = []
    for p in points:
        str_rep = p.get_str_rep()
        if str_rep in seen:
            continue
        seen.add(str_rep)
        ret.append(p)
    return ret

@cached(LRUCache(maxsize=10000))
def collision_detection(p:MeshPoint, 
                        start_p:MeshPoint, 
                        main_dir:Direction, 
                        mesh:Mesh,
                        check_plane_distance=False)-> Union[None, str]:
    # 碰撞检测在真实坐标进行
    x,y,z = p.x/mesh.mesh_num, p.y/mesh.mesh_num, p.z/mesh.mesh_num
    start_x, start_y, start_z = start_p.x/mesh.mesh_num, start_p.y/mesh.mesh_num, start_p.z/mesh.mesh_num
    # 给定坐标和半径，看是否与structure_atoms的任意原子有碰撞
    valid = True
    pt = False
    def distance(x1, y1, z1, x2, y2, z2):
        # print(x1, y1, z1, x2, y2, z2)
        return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))
    
    def plane_distance(x1, y1, z1, x2, y2, z2, dir:Direction):
        # 计算与出发点开始的主方向射线的距离
        # 两组坐标分别为出发点和当前点
        if dir in [Direction.POS_X, Direction.NEG_X]:
            # 计算yz平面距离
            return distance(0, y1, z1, 0, y2, z2)
        elif dir in [Direction.POS_Y, Direction.NEG_Y]:
            # 计算xz平面距离
            return distance(x1, 0, z1, x2, 0, z2)
        elif dir in [Direction.POS_Z, Direction.NEG_Z]:
            # 计算xy平面距离
            return distance(x1, y1, 0, x2, y2, 0)

    for atom in mesh.structure_atoms:
        d = distance(x,y,z, atom[0], atom[1], atom[2])
        if d<mesh.probe_r+mesh.atom_radius[atom[3]]:
            # 原子距离小于两个原子半径的和，则有碰撞
            valid = False
            break
        if atom[3]=='Pt' and mesh.probe_r+mesh.atom_radius[atom[3]]<d<mesh.mesh_radius+mesh.probe_r+mesh.atom_radius[atom[3]]:
            # 对于铂原子，原子距离大于两个原子半径和且小于半径和+网格斜径，则可碰到pt
            pt = True
    if valid:
        if pt:
            if check_plane_distance:
                if plane_distance(x,y,z, start_x, start_y, start_z, dir=main_dir)>=mesh.max_plane_distance:
                    return CollisionState.SEMI_VALID_PT
            return CollisionState.VALID_PT
        else:
            return CollisionState.VALID
    else:
        return CollisionState.INVALID

@cached(LRUCache(maxsize=10000))
def get_points_to_expand(p:MeshPoint, dir:Direction, mesh:Mesh) -> List[MeshPoint]:
    def expand(origin_point:MeshPoint, main_pos, main_value, delta_pos1, delta_pos2):
        # 在主方向上往前进1，与主方向正交的平面上扩展至多probe_mesh_delta
        points = []
        for i in mesh.single_direction_delta:
            for j in mesh.single_direction_delta:
                delta = [0,0,0]
                delta[main_pos] = main_value
                delta[delta_pos1] = i
                delta[delta_pos2] = j
                new_point = mesh.get_mesh_point(
                    x=origin_point.x+delta[0],
                    y=origin_point.y+delta[1],
                    z=origin_point.z+delta[2]
                )
                if new_point.is_coord_valid(mesh.a*mesh.mesh_num, 
                                            mesh.b*mesh.mesh_num, 
                                            mesh.c*mesh.mesh_num):
                    points.append(new_point)
        return dedupe_mesh_points(points)
    # 分别处理六个方向为主方向的情况
    if dir==Direction.POS_X:
        return expand(p, 0, 1, 1, 2)
    elif dir==Direction.NEG_X:
        return expand(p, 0, -1, 1, 2)
    elif dir==Direction.POS_Y:
        return expand(p, 1, 1, 0, 2)
    elif dir==Direction.NEG_Y:
        return expand(p, 1, -1, 0, 2)
    elif dir==Direction.POS_Z:
        return expand(p, 2, 1, 0, 1)
    elif dir==Direction.NEG_Z:
        return expand(p, 2, -1, 0, 1)
    else:
        raise ValueError("wrong direction")

def is_main_dir_distance_valid(start_point:MeshPoint, p:MeshPoint, main_dir:Direction):
    if main_dir==Direction.POS_X:
        return p.x>=start_point.x
    elif main_dir==Direction.NEG_X:
        return p.x<=start_point.x
    elif main_dir==Direction.POS_Y:
        return p.y>=start_point.y
    elif main_dir==Direction.NEG_Y:
        return p.y<=start_point.y
    elif main_dir==Direction.POS_Z:
        return p.z>=start_point.z
    elif main_dir==Direction.NEG_Z:
        return p.z<=start_point.z
    else:
        raise ValueError("unknown direction")

def connected(p1:MeshPoint, p2:MeshPoint):
    # whether there is a path from p2 to p1
    cur = p1
    while True:
        if cur==p2:
            return True
        if cur.father is None:
            return False
        cur = cur.father

def get_start_plane_and_main_direction(p: MeshPoint, mesh:Mesh)-> Tuple[Plane, Direction]:
    # 给定起点坐标，判断是从哪个面出发
    # 立方体的六个面编号见图
    if p.z==0:
        return (Plane.Z_IS_0, Direction.POS_Z)
    if p.z==int(mesh.c*mesh.mesh_num):
        return (Plane.Z_IS_C, Direction.NEG_Z)
    if p.x==0:
        return (Plane.X_IS_0, Direction.POS_X)
    if p.x==int(mesh.a*mesh.mesh_num):
        return (Plane.X_IS_A, Direction.NEG_X)
    if p.y==0:
        return (Plane.Y_IS_0, Direction.POS_Y)
    if p.y==int(mesh.b*mesh.mesh_num):
        return (Plane.Y_IS_B, Direction.NEG_Y)
    return (Plane.NOT_BOUNDARY, Direction.OTHER)
    

def bfs(start_point:MeshPoint, 
        dir:Direction, 
        mesh:Mesh, 
        main_dir:Direction, 
        check_plane_distance=False,
        check_main_dir_distance=False):
    # 用深度优先搜索找到所有可达的网格点
    points:List[MeshPoint] = []
    if start_point.is_coord_valid(a=mesh.a*mesh.mesh_num,
                                  b=mesh.b*mesh.mesh_num,
                                  c=mesh.c*mesh.mesh_num) and start_point.is_collision_state_valid():
        points.append(start_point)
    pointer = 0
    while pointer<len(points):
        origin_point = points[pointer]
        point_to_expands = get_points_to_expand(origin_point, dir, mesh)
        for p in point_to_expands:
            p.collision_state = collision_detection(p, start_point, main_dir, mesh, check_plane_distance)
            if check_main_dir_distance and not is_main_dir_distance_valid(start_point, p, main_dir):
                continue
            if p.is_collision_state_valid():
                if origin_point.collision_state.value>p.collision_state.value:
                    # 这里还要特别考虑一点是将是否能碰到pt这个信息传递下去
                    # 如果这条路更好，要覆盖状态
                    p.collision_state = origin_point.collision_state
                if not mesh.has_visited_before(p) and not connected(origin_point, p):
                    # 找到一个可行的点，将其添加到待扩展队列里
                    p.father = origin_point
                    points.append(p)
                    # 标记这个点，避免重复访问
                    mesh.visit(p)
        # 当前节点扩展完毕
        pointer += 1
    return points   

def get_nearest_mesh_point(p:Point,mesh:Mesh) -> MeshPoint:
    # 输入x,y,z坐标，返回网格格点
    mx = round(p.x*mesh.mesh_num*mesh.a)
    my = round(p.y*mesh.mesh_num*mesh.b)
    mz = round(p.z*mesh.mesh_num*mesh.c)
    return mesh.get_mesh_point(x=mx, y=my, z=mz)

def get_trace(p:MeshPoint):
    trace = []
    trace.append(p)
    cur_node = p
    while cur_node.father is not None:
        cur_node = cur_node.father
        trace.append(cur_node)
    return trace[::-1]

def verify_point(p: Point, mesh:Mesh):
    start_point = get_nearest_mesh_point(p, mesh)
    start_plane, main_dir = get_start_plane_and_main_direction(start_point, mesh)
    start_point.collision_state = collision_detection(start_point, start_point, main_dir, mesh)
    
    all_trace = []
    # 先沿着主方向探索
    all_visited_points:List[MeshPoint] = []
    main_dir_visited_points = bfs(start_point, main_dir, mesh, main_dir, check_plane_distance=True)
    # 判断是否找到合法的终点
    for p in main_dir_visited_points:
        if p.is_valid_end_point(start_plane, mesh):
            all_trace.append(get_trace(p))
            break
    all_visited_points.extend(main_dir_visited_points)
    
    # 分别往与主方向正交的四个方向探索
    for dir in [Direction.NEG_X, Direction.NEG_Y, Direction.NEG_Z,
                Direction.POS_X, Direction.POS_Y, Direction.POS_Z]:
        if len(all_trace)==0: # 如果还没找到合法路径就继续探索
            if abs(dir.value) == abs(main_dir.value):
                # 主方向和主方向的反方向直接放弃
                continue
            # 其他方向应该从主方向所有探索过的且碰到过pt的点开始
            for p in main_dir_visited_points:
                if p.collision_state==CollisionState.VALID_PT:
                    # 第二波探索需要检测点到出发点的主方向距离>=0
                    _visited_points = bfs(p, dir, mesh, main_dir, check_plane_distance=False, check_main_dir_distance=True)
                    all_visited_points.extend(_visited_points)

                    for p in _visited_points:
                        if p.is_valid_end_point(start_plane, mesh):
                            all_trace.append(get_trace(p))
                            break
    if len(all_trace)>0:
        return 'A', all_trace

    # 不是A类，退而求其次，找C 类
    for p in all_visited_points:
        if p.is_semi_valid_end_point(start_plane, mesh):
            all_trace.append(get_trace(p))
            return 'C', all_trace

    mesh.shutdown()
    # output a B type trace
    for p in all_visited_points:
        if p.collision_state in [CollisionState.VALID_PT]:
            if collision_detection(p, start_point, main_dir, mesh, True)==CollisionState.VALID_PT:
                all_trace.append(get_trace(p))
                break   
    return 'B', all_trace

def convert_trace(trace:List[MeshPoint], mesh:Mesh):
    # 把路径转化成归一化坐标点的列表
    ret = []
    for p in trace:
        ret.append((p.x/mesh.mesh_num/mesh.a, 
                    p.y/mesh.mesh_num/mesh.b, 
                    p.z/mesh.mesh_num/mesh.c, str(p.collision_state)))
    return ret

def output_trace(trace:List, save_path):
    with open(save_path, 'w') as f:
        for p in trace:
            f.write(str(p))
            f.write("\n")

if __name__ == '__main__':
    # 给定一个起点坐标，某个目标路径
    # 如果有路径就是A类点
    # 没有路径就是B类点
    parser = ArgumentParser()
    parser.add_argument('--param_path', 
                        type=str, 
                        default='../data/params.json',
                        help="path of params.json")
    # 注意，这里存储的都是归一化坐标
    parser.add_argument('--atom_path', 
                        type=str, 
                        default='../data/new_data.json',
                        help="path of new_data.json")
    args = parser.parse_args()

    with open(args.atom_path, 'r') as f:
        atom_info = json.load(f)

    config = load_config(args.param_path)
    
    for idx, start_point in enumerate(config['start_points']):
        print(f"{'-'*10} {idx} {'-'*10}")
        print(start_point)
        # 每次先新建一个 mesh
        mesh = Mesh(args.param_path, deepcopy(atom_info['structure_atoms']))
        # 验证一个提供的开始点
        hole_type, traces = verify_point(Point(x=start_point[0],
                                               y=start_point[1],
                                               z=start_point[2]),
                                        mesh)
        print('hole type:', hole_type)
        if len(traces)==0:
            move_path = []
        else:
            move_path = convert_trace(traces[0], mesh)
        output_trace(move_path, f'../data/hole_path_{idx}.txt')
        
    