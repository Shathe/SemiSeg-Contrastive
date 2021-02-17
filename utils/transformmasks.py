'''
Code taken from https://github.com/WilhelmT/ClassMix
'''

import torch

'''
def generate_cloud_mask(img_size, sigma, p,seed=None):
    T=10
    np.random.seed(seed)
    # Randomly draw sigma from log-uniform distribution
    N = np.random.normal(size=img_size) # Generate noise image
    Ns = gaussian_filter(N, sigma) # Smooth with a Gaussian
    Ns_norm = (Ns-Ns.mean())/Ns.std()
    Ns_sharp = np.tanh(T*Ns_norm)
    Ns_normalised = (Ns_sharp - np.min(Ns_sharp))/np.ptp(Ns_sharp)
    return Ns_normalised'''

def generate_class_mask(pred, classes): # pred 512,512   classes 9
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N
'''
def generate_cow_class_mask(pred, classes, sigma, p,):
    N=np.zeros(pred.shape)
    pred = np.array(pred.cpu())
    for c in classes:
        N[pred==c] = generate_cow_mask(pred.shape,sigma,p)[pred==c]
    return N'''
