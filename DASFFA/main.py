# -*- encoding: utf-8 -*-
import time
import random
import math
import fire
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from dataset import AmazonData
from models import Model
import methods
import config


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def unpack_input(opt, x):

    uids, iids = list(zip(*x))
    uids = list(uids)
    iids = list(iids)

    user_reviews = opt.user_list[uids]
    user_item2id = opt.user2itemid_dict[uids]  # 检索出该user对应的item id

    user_doc = opt.user_doc[uids]

    item_reviews = opt.item_list[iids]
    item_user2id = opt.item2userid_dict[iids]  # 检索出该item对应的user id
    item_doc = opt.item_doc[iids]

    train_data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    train_data = list(map(lambda x: torch.LongTensor(x).cuda(), train_data))
    return train_data


def train(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Office_Products_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # 2 model
    model = Model(opt, getattr(methods, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if opt.load_ckp:
        assert len(opt.ckp_path) > 0
        model.load(opt.ckp_path)

    # 3 data
    train_data = AmazonData(opt.data_root, train=True)
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)
    test_data = AmazonData(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('{}: train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))

    # 4 optimiezer
    # optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6, weight_decay=opt.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    # optimizer = optim.RMSprop(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=3, after_scheduler=scheduler)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=opt.lr, max_lr=opt.lr * 10)

    # training
    print("start training....")
    min_loss = 1e+20
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print("{} Epoch {}: start".format(now(), epoch))
        for idx, (train_datas, scores) in enumerate(train_data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)
            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()

            smooth_mae_loss = smooth_mae_func(output, scores)

            if opt.update_method == 'mse':
                loss = mse_loss
            if opt.update_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.update_method == 'mae':
                loss = mae_loss
            if opt.update_method == 'smooth_mae':
                loss = smooth_mae_loss

            loss.backward()
            optimizer.step()

            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    predict_loss, test_mse = predict(model, test_data_loader, opt, use_gpu=opt.use_gpu)
                    if predict_loss < min_loss:
                        model.save(name=opt.dataset, opt=opt.print_opt)
                        min_loss = predict_loss
                        print("\tmodel save")
                    if predict_loss > min_loss:
                        best_res = min_loss

        scheduler.step(epoch)
        print("{};epoch:{};total_loss:{}".format(now(), epoch, total_loss))
        mse = total_loss * 1.0 / len(train_data)
        mae = total_maeloss * 1.0 / len(train_data)
        print("{};train reslut: mse: {}; rmse: {}; mae: {}".format(now(), mse, math.sqrt(mse), mae))
        predict_loss, test_mse = predict(model, test_data_loader, opt, use_gpu=opt.use_gpu)
        if predict_loss < min_loss:
            # model.save(name=opt.dataset, opt=opt.print_opt)
            min_loss = predict_loss
            print("model save")
        if test_mse < best_res:
            best_res = test_mse

    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----"*20)


def predict(model, test_data_loader, opt, use_gpu=True):
        total_loss = 0.0
        total_maeloss = 0.0
        model.eval()
        step = 0
        for idx, (test_data, scores) in enumerate(test_data_loader):
            if use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)
            output = model(test_data)
            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()
            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()
            step += len(scores)
        mse = total_loss * 1.0 / step
        mae = total_maeloss * 1.0 / step
        print("\t{};test reslut: mse: {}; rmse: {}; mae: {}".format(now(), mse, math.sqrt(mse), mae))
        model.train()
        return total_loss, mse


if __name__ == "__main__":
    fire.Fire()
