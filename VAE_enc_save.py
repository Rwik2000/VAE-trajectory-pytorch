import numpy as np
import torch
from net_VAE import VAE
train_images = np.load("../Numpy_Dataset/train_data_im_torch.npy")
train_images = train_images/255
_,_,H,W = train_images.shape
encmodel = VAE(H,W,25,1)
encmodel = torch.load("./models/vae_model_30.pt")
encmodel.bs = 1
def SaveEncodings():
    encodings = []
    for i in range(len(train_images)):
        print(i)
        inp = torch.Tensor(train_images[i])
        inp = inp.unsqueeze(0)
        inp = inp.cuda()
        m, var = encmodel.encoder(inp)
        z = encmodel.sampling(m,var)
        z = z.cpu().detach().numpy()
        encodings.append(z[0])
    print(encodings[0])
    np.save("../Numpy_Dataset/train_data_encodings_torch.npy", np.array(encodings))
    print("Saved!")

SaveEncodings()
