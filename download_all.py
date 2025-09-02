import gdown
from global_configs import ckpt_part2_url, ckpt_part3_url, ckpt_part4_url, atkimg_part4_url
import os
import zipfile

part_no = 2
if not os.path.exists(os.path.join('part%d' % part_no, 'resources')):
    os.makedirs(os.path.join('part%d' % part_no, 'resources'))
output_path = os.path.join('part%d' % part_no, 'resources', 'part%d_models.zip' % part_no)
gdown.download(ckpt_part2_url, output_path, quiet=False,fuzzy=True)
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join('part%d' % part_no, 'resources'))
os.remove(output_path)

part_no = 3
if not os.path.exists(os.path.join('part%d' % part_no, 'resources')):
    os.makedirs(os.path.join('part%d' % part_no, 'resources'))
output_path = os.path.join('part%d' % part_no, 'resources', 'part%d_models.zip' % part_no)
gdown.download(ckpt_part3_url, output_path, quiet=False,fuzzy=True)
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join('part%d' % part_no, 'resources'))
os.remove(output_path)

part_no = 4
if not os.path.exists(os.path.join('part%d' % part_no, 'resources')):
    os.makedirs(os.path.join('part%d' % part_no, 'resources'))
output_path = os.path.join('part%d' % part_no, 'resources', 'part%d_models.zip' % part_no)
gdown.download(ckpt_part4_url, output_path, quiet=False,fuzzy=True)
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.join('part%d' % part_no, 'resources'))
os.remove(output_path)

if not os.path.exists(os.path.join('part4_attack')):
    os.makedirs('part4_attack')
output_path = os.path.join('part4_attack', 'adv_dataset.zip')
gdown.download(atkimg_part4_url, output_path, quiet=False,fuzzy=True)
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall('part4_attack')
os.remove(output_path)