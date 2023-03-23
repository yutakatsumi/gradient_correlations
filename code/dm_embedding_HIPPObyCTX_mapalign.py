import subprocess
import numpy as np
import nibabel as nib
from sklearn.metrics import pairwise_distances
import sys
sys.path.append("/cluster/iaslab/gradient/scripts/mapalign-master")
from mapalign import embed

dconn_file = 'HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.HIPPObyCTX.dconn.nii'
dconn = np.tanh(nib.load(dconn_file).get_data())
N = dconn.shape[0]
perc = np.array([np.percentile(x, 90) for x in dconn])
for i in range(dconn.shape[0]):
    dconn[i, dconn[i,:] < perc[i]] = 0
#dconn[dconn < 0] = 0
aff = 1 - pairwise_distances(dconn, metric='cosine')

tmp = nib.load('/cluster/iaslab/gradient/ROIs/cope1_HIPPOonly.dscalar.nii')
tmp_cifti = nib.cifti2.load('/cluster/iaslab/gradient/ROIs/cope1_HIPPOonly.dscalar.nii')

mim = tmp.header.matrix[1]
for idx, bm in enumerate(mim.brain_models):
    print((idx, bm.index_offset, bm.brain_structure))

# normalized angle
import math
aff_na = 1 - (np.arccos(aff) / math.pi)
emb, res = embed.compute_diffusion_map(aff_na, n_components=10, alpha=0.5, return_result=True)
np.save('normangle_affinity_HIPPObyCTX_emb.npy', emb)
np.save('normangle_affinity_HIPPObyCTX_res.npy', res)

for g in range(0,10):
    emb_temporary = emb[:,g]
    emb_temporary = emb_temporary.T
    emb_temporary.shape = (1, len(emb[:,g]))
    img = nib.cifti2.Cifti2Image(emb_temporary, nib.cifti2.Cifti2Header(tmp.header.matrix))
    g_idx = g + 1
    img.to_filename('/cluster/iaslab/gradient/HCP/hippocampus/HarvardOxford-B-hippocampus-thr50-2mm_g' + str(g_idx) + '.HIPPObyCTX.normangle.dscalar.nii')
    cmd = 'wb_command -cifti-separate /cluster/iaslab/gradient/HCP/hippocampus/HarvardOxford-B-hippocampus-thr50-2mm_g' + str(g_idx) + '.HIPPObyCTX.normangle.dscalar.nii COLUMN -volume-all /cluster/iaslab/gradient/HCP/hippocampus/HarvardOxford-B-hippocampus-thr50-2mm_g' + str(g_idx) + '.HIPPObyCTX.normangle.nifti.nii'
    print(cmd)
    subprocess.check_output(cmd, shell=True)

# normalize emb to have a range [0,1]
for g in range(0,10):
    tmpimg = emb[:,g] - np.min(emb[:,g])
    emb[:,g] = np.divide(tmpimg, np.max(tmpimg))
np.save('normangle_affinity_HIPPObyCTX_emb_scaled.npy', emb)

# output gradients (one per file)
for g in range(0,10):
    emb_temporary = emb[:,g]
    emb_temporary = emb_temporary.T
    emb_temporary.shape = (1, len(emb[:,g]))
    img = nib.cifti2.Cifti2Image(emb_temporary, nib.cifti2.Cifti2Header(tmp.header.matrix))
    g_idx = g + 1
    img.to_filename('/cluster/iaslab/gradient/HCP/hippocampus/HarvardOxford-B-hippocampus-thr50-2mm_g' + str(g_idx) + '.HIPPObyCTX.normangle.scaled.dscalar.nii')
    cmd = 'wb_command -cifti-separate /cluster/iaslab/gradient/HCP/hippocampus/HarvardOxford-B-hippocampus-thr50-2mm_g' + str(g_idx) + '.HIPPObyCTX.normangle.scaled.dscalar.nii COLUMN -volume-all /cluster/iaslab/gradient/HCP/hippocampus/HarvardOxford-B-hippocampus-thr50-2mm_g' + str(g_idx) + '.HIPPObyCTX.normangle.scaled.nifti.nii'
    print(cmd)
    subprocess.check_output(cmd, shell=True)