import numpy as np
import os

'''
txt_files = \
    ['J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/AccelScansComponents/Movement/Training/G1EpoxyRasterPlate_Movement_X_train1.txt',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/FricScans/FSR1/Training/G1EpoxyRasterPlate_FrictionFSR1_Movement_train1.txt',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/FricScans/FSR2/Training/G1EpoxyRasterPlate_FrictionFSR2_Movement_train1.txt',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/MetalDetectionScans/Training/G1EpoxyRasterPlate_MetalDetection_Movement_train1.txt',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/ReflectanceScans/Training/G1EpoxyRasterPlate_Reflectance_Movement_train1.txt',
     ]
'''

txt_paths = \
    ['J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/AccelScansComponents/Movement/Training',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/FricScans/FSR1/Training',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/FricScans/FSR2/Training',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/MetalDetectionScans/Training',
    'J:/Dataset/_tactile/TexTUM/LMT_108_SurfaceMaterials_Database/ReflectanceScans/Training',
     ]

keywords = ["_X_", "", "", "", ""]  # 찾고 싶은 문자열로 바꾸세요

file_name_arrays = []
for i, txt_path in enumerate(txt_paths):
    file_name_array = []
    for foldername, subfolders, filenames in os.walk(txt_path):
        for filename in filenames:
            if keywords[i] in filename:
                # file_path = os.path.join(foldername, filename)
                file_name_array.append(filename)

    file_name_arrays.append(file_name_array)

for i in range (0, file_name_arrays[0].__len__()):
    datas = []
    for j, txt_path in enumerate(txt_paths):
        file_name = file_name_arrays[j][i]
        file_path = os.path.join(txt_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        # 데이터를 숫자로 변환 (필요에 따라 float/int 등으로 변환)
        # 예: 모든 줄이 숫자라고 가정할 경우
        data = np.array([float(line) for line in lines])
        datas.append(data)

    datas = np.array(datas)
    datas = np.swapaxes(datas, 0, 1)

    output_name_p1 = os.path.splitext(file_name_arrays[0][i])[0].split('_')[0]
    output_name_p2 = os.path.splitext(file_name_arrays[0][i])[0].split('_')[3]

    np.save('J:/Dataset/_tactile/TexTUM/LMT_108_surface_proc/accum_48K_5/' + output_name_p1 + '_' + output_name_p2 +
            '_accum_48K_5.npy', datas)

'''
for file_name in file_names:
    datas = []
    for txt_path in txt_paths:
        txt_file = os.path.join(txt_path, filename)
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    
        # 데이터를 숫자로 변환 (필요에 따라 float/int 등으로 변환)
        # 예: 모든 줄이 숫자라고 가정할 경우
        data = np.array([float(line) for line in lines])
        datas.append(data)
    
    datas = np.array(datas)
    datas = np.swapaxes(datas, 0, 1)
    print(datas.shape)

    np.save('J:/Dataset/_tactile/TexTUM/LMT_108_surface_proc/' + os.path.splitext(filename[0])[0] + 'accum_amffmr.npy', datas)
    exit(0)
'''