import subprocess
import numpy as np
import nibabel as nib
from sklearn.metrics import pairwise_distances
import sys
sys.path.append("/cluster/iaslab/gradient/scripts/mapalign-master")
from mapalign import embed

dconn = np.tanh(nib.load('HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.CTXbyCTX.dconn.nii').get_data())
N = dconn.shape[0]
perc = np.array([np.percentile(x, 90) for x in dconn])
for i in range(dconn.shape[0]):
    #print("Row %d" % i)
    dconn[i, dconn[i,:] < perc[i]] = 0
dconn[dconn < 0] = 0
aff = 1 - pairwise_distances(dconn, metric='cosine')
np.save('cosine_affinity_CTXbyCTX.npy', aff)
aff = np.load('cosine_affinity_CTXbyCTX.npy')
emb, res = embed.compute_diffusion_map(aff, n_components=10, alpha=0.5, return_result=True)
np.save('cosine_affinity_CTXbyCTX_emb.npy', emb)
np.save('cosine_affinity_CTXbyCTX_res.npy', res)
emb = np.load('cosine_affinity_CTXbyCTX_emb.npy')

tmp = nib.nifti2.load('/autofs/cluster/iaslab/gradient/ROIs/100307_tfMRI_MOTOR_level2_hp200_s2.cortex_only.dscalar.nii')
tmp_cifti = nib.cifti2.load('/autofs/cluster/iaslab/gradient/ROIs/100307_tfMRI_MOTOR_level2_hp200_s2.cortex_only.dscalar.nii')
data = tmp_cifti.get_data() * 0
data[0:10,:len(emb)] = np.reshape(emb.T, [1, 1, 1, 1] + list(emb.T.shape))
img = nib.cifti2.Cifti2Image(data, nib.cifti2.Cifti2Header(tmp_cifti.header.matrix))
img.to_filename('/cluster/iaslab/gradient/HCP/cortex/CTX2CTX.dscalar.nii')