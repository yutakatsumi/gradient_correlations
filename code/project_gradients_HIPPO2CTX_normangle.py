import numpy as np
import nibabel as nib
import scipy.stats as st

# dconn here should always be ROIbyCTX
dconn_file = 'HCP_S1200_1003_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.HIPPObyCTX.dconn.nii'
dconn = nib.load(dconn_file).get_fdata()
emb = np.load('normangle_affinity_HIPPObyCTX_emb.npy')
#emb.shape = (1033,10)
emb_g1 = emb[:,1]
# flip sign
#emb_g1 = emb_g1*-1

weighted_dconn = np.dot(emb_g1,dconn)
#weighted_dconn.shape = (59412,)

tmp = nib.load('/cluster/iaslab/gradient/ROIs/100307_tfMRI_MOTOR_level2_hp200_s2.cortex_only.dscalar.nii')
tmp_cifti = nib.cifti2.load('/cluster/iaslab/gradient/ROIs/100307_tfMRI_MOTOR_level2_hp200_s2.cortex_only.dscalar.nii')
data = tmp_cifti.get_data() * 0
data[0,:] = weighted_dconn
img = nib.cifti2.Cifti2Image(data, nib.cifti2.Cifti2Header(tmp.header.matrix))
img.to_filename('HIPPObyCTX.weighted_connectivity_CTX-LR.g2.normangle.dscalar.nii')


# zxfm G3
emb_g3 = emb[:,2]
emb_g3[0:511,] = st.zscore(emb_g3[0:511,])
emb_g3[511:,] = st.zscore(emb_g3[511:,])

weighted_dconn = np.dot(emb_g3,dconn)
data = tmp_cifti.get_data() * 0
data[0,:] = weighted_dconn
img = nib.cifti2.Cifti2Image(data, nib.cifti2.Cifti2Header(tmp.header.matrix))
img.to_filename('HIPPObyCTX.weighted_connectivity_CTX-LR.g3.normangle.dscalar.nii')