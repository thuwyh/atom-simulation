from copy import deepcopy
import json
structure_atoms = []
with open('../data/atom_pos.txt', 'r') as f:
    for line in f:
        x,y,z = line.split()
        # 转化成真实坐标
        structure_atoms.append([float(x), float(y), float(z), 'Si'])

# 主要是要去掉这些手写的数字
assert len(structure_atoms)==384+768+13+6
for i in range(384, 384+768):
    structure_atoms[i][3]='O'
for i in range(384+768, 384+768+13):
    structure_atoms[i][3]='Pt'
start_points = []
for x in structure_atoms[-6:]:
    start_points.append([x[0], x[1], x[2]])
structure_atoms = structure_atoms[:-6] # 去掉6个开始点

with open("../data/new_data.json", 'w') as f:
    json.dump({
        'structure_atoms': structure_atoms,
    }, f, indent=2, ensure_ascii=False)

delta_x=0
delta_y=0
delta_z=0
nx=2
ny=2
nz=2
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            new_structure_atoms = deepcopy(structure_atoms)
            for idx in range(len(new_structure_atoms)):
                if new_structure_atoms[idx][3]=='Pt':
                    new_structure_atoms[idx][0]+=i*delta_x
                    new_structure_atoms[idx][1]+=j*delta_y
                    new_structure_atoms[idx][2]+=k*delta_z
            # save new file
            with open(f"../data/new_data.json_{i}{j}{k}", 'w') as f:
                json.dump({
                    f'structure_atoms': new_structure_atoms,
                }, f, indent=2, ensure_ascii=False)
