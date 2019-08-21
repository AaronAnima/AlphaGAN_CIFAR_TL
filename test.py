import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags
from data import get_dataset_train, get_dataset_eval
from models import get_G, get_img_D, get_E, get_z_D
import random
import argparse
import math


def ktest():
    print('Start ktest!')
    ds, _ = get_dataset_train()
    E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E.load_weights('{}/{}/E.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
    E.eval()


def disentangle_test():
    print('Start disentangle_test!')
    ds, _ = get_dataset_train()
    E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E.load_weights('{}/{}/E.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
    E.eval()
    G = get_G([None, flags.z_dim])
    G.load_weights('{}/{}/G.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
    G.eval()
    for step, batch_img in enumerate(ds):
        if step > flags.disentangle_step_num:
            break
        z_real = E(batch_img)
        hash_real = ((tf.sign(z_real * 2 - 1, name=None) + 1)/2).numpy()
        epsilon = flags.scale * np.random.normal(loc=0.0, scale=flags.sigma * math.sqrt(flags.z_dim) * 0.0625,
                                           size=[flags.batch_size_train, flags.z_dim]).astype(np.float32)
        # z_fake = hash_real + epsilon
        z_fake = z_real + epsilon
        fake_imgs = G(z_fake)
        tl.visualize.save_images(batch_img.numpy(), [8, 8],
                                 '{}/{}/disentangle/real_{:02d}.png'.format(flags.test_dir, flags.param_dir, step))
        tl.visualize.save_images(fake_imgs.numpy(), [8, 8],
                                 '{}/{}/disentangle/fake_{:02d}.png'.format(flags.test_dir, flags.param_dir, step))


def Evaluate_mAP():

    ####################### Functions ################

    class Retrival_Obj():
        def __init__(self, hash, label):
            self.label = label
            self.dist = 0
            list1 = [True if hash[i] == 1 else False for i in range(len(hash))]
            # convert bool list to bool array
            self.hash = np.array(list1)

        def __repr__(self):
            return repr((self.hash, self.label, self.dist))

    # to calculate the hamming dist between obj1 & obj2

    def hamming(obj1, obj2):
        res = obj1.hash ^ obj2.hash
        ans = 0
        for k in range(len(res)):
            if res[k] == True:
                ans += 1
        obj2.dist = ans

    def take_ele(obj):
        return obj.dist

    # to get 'nearest_num' nearest objs from 'image' in 'Gallery'
    def get_nearest(image, Gallery, nearest_num):
        for obj in Gallery:
            hamming(image, obj)
        Gallery.sort(key=take_ele)
        ans = []
        cnt = 0
        for obj in Gallery:
            cnt += 1
            if cnt <= nearest_num:
                ans.append(obj)
            else:
                break

        return ans

    # given retrivial_set, calc AP w.r.t. given label
    def calc_ap(retrivial_set, label):
        total_num = 0
        ac_num = 0
        ans = 0
        result = []
        for obj in retrivial_set:
            total_num += 1
            if obj.label == label:
                ac_num += 1
            ans += ac_num / total_num
            result.append(ac_num / total_num)
        result = np.array(result)
        ans = np.mean(result)
        return ans

    ################ Start eval ##########################

    print('Start Eval!')
    # load images & labels
    ds, _= get_dataset_train()
    E = get_E([None, flags.img_size_h, flags.img_size_w, flags.c_dim])
    E.load_weights('{}/{}/E.npz'.format(flags.checkpoint_dir, flags.param_dir), format='npz')
    E.eval()

    # create (hash,label) gallery
    Gallery = []
    cnt = 0
    step_time1 = time.time()
    for batch, label in ds:
        cnt += 1
        if cnt % flags.eval_print_freq == 0:
            step_time2 = time.time()
            print("Now {} Imgs done, takes {:.3f} sec".format(cnt, step_time2 - step_time1))
            step_time1 = time.time()
        hash_fake, _ = E(batch)
        hash_fake = hash_fake.numpy()[0]
        hash_fake = ((tf.sign(hash_fake*2 - 1, name=None) + 1)/2).numpy()
        label = label.numpy()[0]
        Gallery.append(Retrival_Obj(hash_fake, label))
    print('Hash calc done, start split dataset')

    #sample 1000 from Gallery and bulid the Query set
    random.shuffle(Gallery)
    cnt = 0
    Queryset = []
    G = []
    for obj in Gallery:
        cnt += 1
        if cnt > flags.eval_sample:
            G.append(obj)
        else:
            Queryset.append(obj)
    Gallery = G
    print('split done, start eval')

    # Calculate mAP
    Final_mAP = 0
    step_time1 = time.time()
    for eval_epoch in range(flags.eval_epoch_num):
        result_list = []
        cnt = 0
        for obj in Queryset:
            cnt += 1
            if cnt % flags.retrieval_print_freq == 0:
                step_time2 = time.time()
                print("Now Steps {} done, takes {:.3f} sec".format(eval_epoch, cnt, step_time2 - step_time1))
                step_time1 = time.time()

            retrivial_set = get_nearest(obj, Gallery, flags.nearest_num)
            result = calc_ap(retrivial_set, obj.label)
            result_list.append(result)
        result_list = np.array(result_list)
        temp_res = np.mean(result_list)
        print("Query_num:{}, Eval_step:{}, Top_k_num:{}, AP:{:.3f}".format(flags.eval_sample, eval_epoch,
                                                                           flags.nearest_num, temp_res))
        Final_mAP += temp_res / flags.eval_epoch_num
    print('')
    print("Query_num:{}, Eval_num:{}, Top_k_num:{}, mAP:{:.3f}".format(flags.eval_sample, flags.eval_epoch_num,
                                                                    flags.nearest_num, Final_mAP))
    print('')


def Evaluate_Cluster():
    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mAP', help='train or eval')

    args = parser.parse_args()

    print(args.mode)
    if args.mode == 'mAP':
        Evaluate_mAP()
    elif args.mode == 'cluster':
        Evaluate_Cluster()
    elif args.mode == 'disentangle':
        tl.files.exists_or_mkdir(flags.test_dir + '/' + flags.param_dir + '/' + 'disentangle')  # test save path
        disentangle_test()
    else:
        raise Exception("Unknow --mode")
