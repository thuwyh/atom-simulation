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
        'start_points': start_points
    }, f, indent=2, ensure_ascii=False)
