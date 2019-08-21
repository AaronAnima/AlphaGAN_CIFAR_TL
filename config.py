import numpy as np
import tensorlayer as tl

class FLAGS(object):
    def __init__(self):
        ''' For training'''
        self.epsilon = 0.001
        self.n_epoch = 100 # "Epoch to train [25]"
        self.z_dim = 64 # "Dim of noise value]"
        self.c_dim = 3 # "Number of image channels. [3]")
        # Learning rate
        self.lr_G = 0.0001
        self.lr_E = 0.0005
        self.lr_D = 0.0005
        self.lr_Dz = 0.0005
        self.lr_Dh = 0.0005
        self.beta1 = 0.5 # "Momentum term of adam [0.5]")
        self.beta2 = 0.9
        self.batch_size_train = 64 # "The number of batch images [64]")
        self.dataset = "CIFAR" # "The name of dataset [CIFAR_10, MNIST]")
        self.checkpoint_dir = "./checkpoint"
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        self.img_size_h = 32 # Img height
        self.img_size_w = 32  # Img width
        self.eval_step = 100 # Evaluation freq during training
        self.lamba_recon = 20
        self.Aug = 0
        self.mode = 0
        self.len_dataset = 60000
        self.step_num = 200000
        self.sigma = 1/4
        self.scale = 1
        self.param_dir ='r' + str(int(1 / self.sigma)) + '_' + 's' + str(self.scale) + '_' + \
                    'd' + str(self.z_dim) + '_' + 'con' + str(self.lamba_recon)
        self.decay = 1
        ''' For eval '''
        self.show_freq = 10
        self.eval_epoch_num = 10
        self.eval_print_freq = 5000 #
        self.retrieval_print_freq = 200
        self.eval_sample = 1000 # Query num for mAP matrix
        self.nearest_num = 1000 # nearest obj num for each query
        self.batch_size_eval = 1  # batch size for every eval
        self.test_dir = './test_results'
        self.disentangle_step_num = 10



flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir + '/' + flags.param_dir)  # checkpoint path
tl.files.exists_or_mkdir(flags.sample_dir + '/' + flags.param_dir)  # samples path



tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image