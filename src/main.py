# 参数
from __future__ import annotations
from typing import List, Optional, Tuple, Union
import math
from pydantic import BaseModel
from enum import Enum
from tqdm import tqdm
import sys
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
    VALID = 0 # 可行点，没碰到任何结构原子
    VALID_PT = 1 # 除了pt没碰到其他任何结构原子
    INVALID = 2 # 会碰到处PT外的其他结构原子


class MeshPoint(BaseModel):
    x: int
    y: int
    z: int
    collision_state: Optional[str]=None
    real_x: Optional[float]=None
    real_y: Optional[float]=None
    real_z: Optional[float]=None
    father:MeshPoint=None
    valid_end_point:bool=None

    def model_post_init(self, __context):
        self.real_x = self.x/mesh_num
        self.real_y = self.y/mesh_num
        self.real_z = self.z/mesh_num


    def is_coord_valid(self, a, b, c):
        if self.x<0 or self.x>a:
            return False
        if self.y<0 or self.y>b:
            return False
        if self.z<0 or self.z>c:
            return False
        return True
    
    def is_collision_state_valid(self, probe_r):
        if self.collision_state is None:
            self.collision_state = collision_detection(self, probe_r)
        if self.collision_state!=CollisionState.INVALID:
            return True
        return False

    def met_with_pt(self):
        if self.collision_state is None:
            self.collision_state = collision_detection(self, probe_r)
        if self.collision_state==CollisionState.VALID_PT:
            return True
        return False
    
    def is_valid_end_point(self, start_plane):
        if self.valid_end_point is not None:
            return self.valid_end_point
        plane, _ = get_start_plane_and_main_direction(self)
        if plane!=Plane.NOT_BOUNDARY and plane!=start_plane and self.collision_state==CollisionState.VALID_PT:
            self.valid_end_point = True
        else:
            self.valid_end_point = False
        return self.valid_end_point
    
    def get_str_rep(self):
        return f"{self.x}_{self.y}_{self.z}"

class Mesh:

    def __init__(self):
        self.visited_point = set()
        self.all_points = {}
        self.pbar = tqdm(total=int(a*b*c*mesh_num*mesh_num*mesh_num),
                         desc="visited points")

    def visit(self, p:MeshPoint):
        self.visited_point.add(p.get_str_rep())
    
    def has_visited_before(self, p:MeshPoint):
        return p.get_str_rep() in self.visited_point
    
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
a = 40.1399993896 # x
b = 19.9200000763 # y
c = 26.8400001526 # z

probe_r = 1.5 # 在一个方向上前进时可以往外扩散的半径
mesh_num = 10 # 一个长度分多少个网格
mesh_radius = math.sqrt(3)/mesh_num # 真实坐标半径
probe_mesh_delta = math.floor(math.sqrt(probe_r*mesh_num)) # 网格坐标下在主方向正交平面内横向探索的delta
# 先分解成单个方向上的范围，方便后面是用。这样处理后正交平面探索范围是个方形
single_direction_delta = list(range(-probe_mesh_delta, probe_mesh_delta+1)) 

atom_radius = {
    'O': 1.4,
    'Si': 2.1,
    'Pt': 2.1
}


structure_atoms = [
    # (0.1, 0.1, 0.1, 'O'), # x,y,z,type
    # (0.2, 0.2, 0.2, 'Si'),
    # (0.3, 0.3, 0.3, 'Pt')
]

# 读数据
with open('../data/atom_pos.txt', 'r') as f:
    for line in f:
        x,y,z = line.split()
        # 转化成真实坐标
        structure_atoms.append([float(x)*a, float(y)*b, float(z)*c, 'Si'])
assert len(structure_atoms)==384+768+13+6
for i in range(384, 384+768):
    structure_atoms[i][3]='O'
for i in range(384+768, 384+768+13):
    structure_atoms[i][3]='Pt'
structure_atoms = structure_atoms[:-6] # 去掉6个开始点

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

def collision_detection(p:MeshPoint,r)-> Union[None, str]:
    # 碰撞检测在真实坐标进行
    x,y,z = p.x/mesh_num, p.y/mesh_num, p.z/mesh_num
    # 给定坐标和半径，看是否与structure_atoms的任意原子有碰撞
    valid = True
    pt = False
    def distance(x1, y1, z1, x2, y2, z2):
        # print(x1, y1, z1, x2, y2, z2)
        return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))

    for atom in structure_atoms:
        d = distance(x,y,z, atom[0], atom[1], atom[2])
        if d<r+atom_radius[atom[3]]:
            # 原子距离小于两个原子半径的和，则有碰撞
            valid = False
            break
        if atom[3]=='Pt' and r+atom_radius[atom[3]]<d<mesh_radius+r+atom_radius[atom[3]]:
            # 对于铂原子，原子距离大于两个原子半径和且小于半径和+网格斜径，则可碰到pt
            pt = True
    if valid:
        if pt:
            return CollisionState.VALID_PT
        else:
            return CollisionState.VALID
    else:
        return CollisionState.INVALID

def get_points_to_expand(p:MeshPoint, dir:Direction, mesh:Mesh) -> List[MeshPoint]:
    def expand(origin_point:MeshPoint, main_pos, main_value, delta_pos1, delta_pos2):
        # 在主方向上往前进1，与主方向正交的平面上扩展至多probe_mesh_delta
        points = []
        for i in single_direction_delta:
            for j in single_direction_delta:
                delta = [0,0,0]
                delta[main_pos] = main_value
                delta[delta_pos1] = i
                delta[delta_pos2] = j
                new_point = mesh.get_mesh_point(
                    x = origin_point.x+delta[0],
                    y=origin_point.y+delta[1],
                    z=origin_point.z+delta[2]
                )
                if new_point.is_coord_valid(a*mesh_num, b*mesh_num, c*mesh_num):
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


def get_start_plane_and_main_direction(p: MeshPoint)-> Tuple[Plane, Direction]:
    # 给定起点坐标，判断是从哪个面出发
    # 立方体的六个面编号见图
    if p.z==0:
        return (Plane.Z_IS_0, Direction.POS_Z)
    if p.z==int(c*mesh_num):
        return (Plane.Z_IS_C, Direction.NEG_Z)
    if p.x==0:
        return (Plane.X_IS_0, Direction.POS_X)
    if p.x==int(a*mesh_num):
        return (Plane.X_IS_A, Direction.NEG_X)
    if p.y==0:
        return (Plane.Y_IS_0, Direction.POS_Y)
    if p.y==int(b*mesh_num):
        return (Plane.Y_IS_B, Direction.NEG_Y)
    return (Plane.NOT_BOUNDARY, Direction.OTHER)
    
def pass_one_direction(start_point: MeshPoint, dir:Direction):
    # 第一遍扩展，只朝主方向前进
    pass

def pass_two(start_plane):
    # 第二遍扩展，朝正交于主方向的平面扩展
    pass

def bfs(start_point:MeshPoint, dir:Direction, mesh:Mesh):
    # 用深度优先搜索找到所有可达的网格点
    points:List[MeshPoint] = []
    if start_point.is_coord_valid(a=a*mesh_num,b=b*mesh_num,c=c*mesh_num) and start_point.is_collision_state_valid(probe_r):
        points.append(start_point)
    pointer = 0
    while pointer<len(points):
        origin_point = points[pointer]
        point_to_expands = get_points_to_expand(origin_point, dir, mesh)
        for p in point_to_expands:
            if not mesh.has_visited_before(p) and p.is_collision_state_valid(probe_r):
                # 找到一个可行的点，将其添加到待扩展队列里
                # 这里还要特别考虑一点是将是否能碰到pt这个信息传递下去
                if origin_point.met_with_pt():
                    p.collision_state = CollisionState.VALID_PT
                p.father = origin_point
                points.append(p)
                # 标记这个点，避免重复访问
                mesh.visit(p)
        # 当前节点扩展完毕
        pointer += 1
    return points   

def get_nearest_mesh_point(p:Point,mesh:Mesh) -> MeshPoint:
    # 输入x,y,z坐标，返回网格格点
    mx = round(p.x*mesh_num*a)
    my = round(p.y*mesh_num*b)
    mz = round(p.z*mesh_num*c)
    return mesh.get_mesh_point(x=mx, y=my, z=mz)

def get_trace(p:MeshPoint):
    if p.father is not None:
        return get_trace(p.father)+[p]
    return [p]

def verify_point(p: Point):
    # 先新建网格
    mesh = Mesh()
    start_point = get_nearest_mesh_point(p, mesh)
    start_point.collision_state = collision_detection(start_point, probe_r)
    start_plane, main_dir = get_start_plane_and_main_direction(start_point)

    all_trace = []
    # 先沿着主方向探索
    all_visited_points = []
    main_dir_visited_points = bfs(start_point, main_dir, mesh)
    # 判断是否找到合法的终点
    for p in main_dir_visited_points:
        if p.is_valid_end_point(start_plane):
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
                    _visited_points = bfs(p, dir, mesh)
                    all_visited_points.extend(_visited_points)

                    for p in _visited_points:
                        if p.is_valid_end_point(start_plane):
                            all_trace.append(get_trace(p))
                            break
    if len(all_trace)>0:
        return 'A', all_trace

    # 不是A类，退而求其次，找碰到过PT的
    for p in all_visited_points:
        if p.collision_state==CollisionState.VALID_PT:
            all_trace.append(get_trace(p))
            break

    mesh.shutdown()    
    return 'B', all_trace

def convert_trace(trace:List[MeshPoint]):
    # 把路径转化成归一化坐标点的列表
    ret = []
    for p in trace:
        ret.append((p.x/mesh_num/a, p.y/mesh_num/b, p.z/mesh_num/c, str(p.collision_state)))
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

    # 0.749700010         0.930790007         0.506309986
    # A类孔，横向探索半径1.5A
    hole_type, traces = verify_point(Point(x=0.749700010,y=1,z=0.506309986))
    print(len(traces))
    move_path = convert_trace(traces[0])
    output_trace(move_path, '../data/hole1_path.txt')

    # 0.497590005         0.926360011         0.760890007
    # hole_type, traces = verify_point(Point(x=0.497590005,y=1,z=0.760890007))
    # print(len(traces))

    # 0.498459995         0.044310000         0.767949998
    # A类孔，横向探索半径1.5A
    # hole_type, traces = verify_point(Point(x=0.498459995,y=0,z=0.767949998))
    # print(len(traces))

    # 0.746129990         0.041219998         0.504649997
    # A类孔，横向探索半径1.5A
    # hole_type, traces = verify_point(Point(x=0.746129990,y=0,z=0.504649997))
    # print(len(traces))

    # 0.988179982         0.736280024         0.600510001
    # hole_type, traces = verify_point(Point(x=1,y=0.736280024,z=0.600510001))
    # print(len(traces))

    # 0.012410000         0.745670021         0.618990004
    # hole_type, traces = verify_point(Point(x=0,y=0.745670021,z=0.618990004))
    # print(len(traces))