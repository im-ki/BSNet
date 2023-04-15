import torch
import random
import pickle
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class QCM_Gen(Dataset):
    def __init__(self, dataset_file_path, output_size):
        
        self.output_size = output_size
        if dataset_file_path[-3:] == 'pkl':
            self.read_image_from_pkl(dataset_file_path)
        elif dataset_file_path[-3:] == 'txt':
            self.read_image_from_txt(dataset_file_path)

        self.img_pipeline = transforms.Compose([transforms.ToPILImage(),
                                                transforms.RandomResizedCrop(output_size, interpolation = transforms.InterpolationMode.BILINEAR),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.GaussianBlur(5, sigma=(0.1, 1.0)),
                                                transforms.RandomInvert(),
                                                transforms.PILToTensor()])
        self.noise_pipeline = transforms.Compose([transforms.Resize(output_size, interpolation = transforms.InterpolationMode.BILINEAR),
                                                  transforms.GaussianBlur(5, sigma=(0.1, 1.0))])
        self.H_coarse = 30
        self.W_coarse = 30

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        mu_coarse_norm = torch.rand(self.H_coarse, self.W_coarse)
        mu_coarse_angle = torch.rand(self.H_coarse, self.W_coarse) * 2 * torch.pi
        mu_coarse = mu_coarse_norm * torch.exp(1j * mu_coarse_angle)
        mu_fine_real = self.noise_pipeline(mu_coarse.real.unsqueeze(0))
        mu_fine_imag = self.noise_pipeline(mu_coarse.imag.unsqueeze(0))
        mu_fine = mu_fine_real + 1j * mu_fine_imag
        mu_fine_mod = torch.abs(mu_fine)
        mu_fine_ang = torch.angle(mu_fine)

        real = self.img_pipeline(self.images[idx]) * random.random()
        imag = self.img_pipeline(self.images[random.randint(0, self.num_sample - 1)]) * random.random()
        mu = real + 1j * imag

        mod = torch.abs(mu)
        max_mod = torch.max(mod)
        mod = mod + mu_fine_mod * max_mod * 0.1
        max_mod = torch.max(mod)
        mod = mod / (max_mod + 1e-8) * random.random()

        ang = torch.angle(mu)
        max_ang = torch.max(ang)
        min_ang = torch.min(ang)
        theta = max_ang - min_ang + 1e-8

        mu = mod * torch.exp(1j * ((2 * torch.pi / theta) * ang + mu_fine_ang))
        mu = torch.concat((mu.real, mu.imag), dim = 0)
  
        return mu

    def read_image_from_pkl(self, path):
        with open(path, 'rb') as f:
            self.images = pickle.load(f)
        self.num_sample = len(self.images)

    def read_image_from_txt(self, path):
        paths = []
        # Read the dataset file
        dataset_file = open(path)
        lines = dataset_file.readlines()
        for line in lines:
            items = line.split()
            paths.append(items[0])

        self.images = []
        for i in range(len(paths)):
            img = torchvision.io.read_image(paths[i])#cv2.imread(paths[i])
            #images.append(cv2.resize(img[:, :, i], (256, 256)).astype(np.float32))
            self.images.append(img[:, :, 0], img[:, :, 1], img[:, :, 2])
        self.num_sample = len(self.images)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.qc_cpu import mu2map
    from utils.show_result import plot_map
    size = [112, 112]
    gen = QCM_Gen('../imagenet.pkl', size)
    for i in range(100):
        a = gen[i]
        mapping = mu2map(a[0])[0]
        plot_map(mapping)

        #print(a.shape, a.dtype, gen.images[0].shape)
        #plt.imshow(a[0], cmap = 'gray')
        #plt.show()
 
