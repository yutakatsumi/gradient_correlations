import subprocess
import numpy as np
import nibabel as nib
from sklearn.metrics import pairwise_distances
import sys
sys.path.append("/cluster/iaslab/gradient/scripts/mapalign-master")
from mapalign import embed

dconn = np.tanh(nib.load('HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.CBbyCTX.dconn.nii').get_data())
N = dconn.shape[0]
perc = np.array([np.percentile(x, 90) for x in dconn])
for i in range(dconn.shape[0]):
    #print("Row %d" % i)
    dconn[i, dconn[i,:] < perc[i]] = 0
dconn[dconn < 0] = 0
aff = 1 - pairwise_distances(dconn, metric='cosine')
np.save('cosine_affinity_CBbyCTX.npy', aff)
aff = np.load('cosine_affinity_CBbyCTX.npy')
emb, res = embed.compute_diffusion_map(aff, n_components=10, alpha=0.5, return_result=True)
np.save('cosine_affinity_CBbyCTX_emb.npy', emb)
np.save('cosine_affinity_CBbyCTX_res.npy', res)
emb = np.load('cosine_affinity_CBbyCTX_emb.npy')

tmp = nib.load('/cluster/iaslab/gradient/ROIs/cope1_Cb_only.dscalar.nii')
tmp_cifti = nib.cifti2.load('/cluster/iaslab/gradient/ROIs/cope1_Cb_only.dscalar.nii')

mim = tmp.header.matrix[1]
for idx, bm in enumerate(mim.brain_models):
    print((idx, bm.index_offset, bm.brain_structure))

for g in range(0,10):
    emb_temporary = emb[:,g]
    emb_temporary = emb_temporary.T
    emb_temporary.shape = (1, len(emb[:,g]))
    img = nib.cifti2.Cifti2Image(emb_temporary, nib.cifti2.Cifti2Header(tmp.header.matrix))
    g_idx = g + 1
    img.to_filename('/cluster/iaslab/gradient/HCP/cerebellum/Cerebellum-MNIfnirt-maxprob-thr50-2mm_g' + str(g_idx) + '.CBbyCTX.cosine.dscalar.nii')
    cmd = 'wb_command -cifti-separate /cluster/iaslab/gradient/HCP/cerebellum/Cerebellum-MNIfnirt-maxprob-thr50-2mm_g' + str(g_idx) + '.CBbyCTX.cosine.dscalar.nii COLUMN -volume-all /cluster/iaslab/gradient/HCP/cerebellum/Cerebellum-MNIfnirt-maxprob-thr50-2mm_g' + str(g_idx) + '.CBbyCTX.cosine.nifti.nii'
    print(cmd)
    subprocess.check_output(cmd, shell=True)