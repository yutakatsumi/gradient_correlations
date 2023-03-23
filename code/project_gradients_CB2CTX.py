import numpy as np
import nibabel as nib

# dconn here should always be ROIbyCTX
dconn_file = 'HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.CBbyCTX.dconn.nii'
dconn = nib.load(dconn_file).get_data()
emb = np.load('cosine_affinity_CBbyCTX_emb.npy')
#emb.shape = (1033,10)
emb_grad = emb[:,2]
# flip sign
#emb_g1 = emb_g1*-1

weighted_dconn = np.dot(emb_grad,dconn)
#weighted_dconn.shape = (59412,)

tmp = nib.load('/cluster/iaslab/gradient/ROIs/100307_tfMRI_MOTOR_level2_hp200_s2.cortex_only.dscalar.nii')
tmp_cifti = nib.cifti2.load('/cluster/iaslab/gradient/ROIs/100307_tfMRI_MOTOR_level2_hp200_s2.cortex_only.dscalar.nii')
data = tmp_cifti.get_data() * 0
data[0,:] = weighted_dconn
img = nib.cifti2.Cifti2Image(data, nib.cifti2.Cifti2Header(tmp.header.matrix))
img.to_filename('CBbyCTX.weighted_connectivity_CTX-LR.g3.cosine.dscalar.nii')
