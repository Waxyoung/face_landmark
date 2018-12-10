import sys
import os
import torch.optim as optim
import torch
import collections
import numpy as np
import math
from multiprocessing import cpu_count
from base_options import BaseOptions
from my_face_dataset import FaceDataSet

SymbolWithOptimizer = collections.namedtuple('SymbolWithOptimizer', ['symbol', 'optimizer'])

if __name__=="__main__":
    opt = BaseOptions().parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    num_workers = cpu_count() if torch.cuda.is_available() else 0

    criterion = torch.nn.MSELoss()

    symbol_list = []
    from importlib import import_module
    for symbol_file_name in opt.symbols:
        symbol_module = import_module('symbols.' + symbol_file_name)
        symbol = symbol_module.get_symbol()
        if opt.load_epoch:
            mode_ends = '_latest.pth' if opt.load_epoch == 'latest' else '%05d.pth'%opt.load_epoch
            checkpoint_path = os.path.join(opt.checkpoints_dir, symbol_file_name + mode_ends)
            if os.path.exists(checkpoint_path): symbol_module.load_model(symbol, checkpoint_path)
        symbol.to(device)
        symbol_optimizer = optim.Adam([ {'params': symbol.conv_block.parameters()}, {'params': symbol.landmark_block.parameters()}], lr = opt.lr)
        symbol_list.append(SymbolWithOptimizer(symbol = symbol, optimizer = symbol_optimizer))

    trainset = FaceDataSet('../trainData', opt.load_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = num_workers, drop_last = True)

    testset = FaceDataSet('../testData', opt.load_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size = opt.batch_size, shuffle = True, num_workers = num_workers, drop_last = True)

    loss_dic = {}
    for epoch in range(5000):
        for symbol_with_optimer in symbol_list:            
            symbol_with_optimer.symbol.train()
            loss_dic[symbol_with_optimer.symbol.op_name] = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data['img'], data['label']
            # print 'inputs:',inputs.shape
            # print 'label:' ,labels.shape
            inputs, labels = inputs.to(device), labels.to(device)

            for symbol_with_optimer in symbol_list:
                
                symbol_with_optimer.optimizer.zero_grad()
                outputs, track_probe = symbol_with_optimer.symbol(inputs)
                loss1=criterion(outputs[:44], labels[:44])
                print 'loss1:',loss1
                loss2=criterion(outputs[44:], labels[44:])
                print 'loss2:',loss2
                loss=0.73333*loss1+0.26667*loss2
                print 'loss:',loss
                # loss = criterion(outputs, labels)
                loss.backward()
                symbol_with_optimer.optimizer.step()
                loss_dic[symbol_with_optimer.symbol.op_name] += loss.item()

            if i % 30 == 29:
                result_info = '[epoch:%d, iter:%5d]'% (epoch + 1, i + 1)
                for symbol_with_optimer in symbol_list:
                    result_info += ' ' + symbol_with_optimer.symbol.op_name + ': %.5f' % math.sqrt(loss_dic[symbol_with_optimer.symbol.op_name] / 30.0)
                    loss_dic[symbol_with_optimer.symbol.op_name] = 0.0
                print result_info
                sys.stdout.flush()
        
        #save the model
        if epoch % 50 == 49:
            for symbol_with_optimer in symbol_list:
                torch.save(symbol_with_optimer.symbol.state_dict(), os.path.join(opt.checkpoints_dir, symbol_with_optimer.symbol.op_name + '%05d.pth'%(epoch + 1)) )
                torch.save(symbol_with_optimer.symbol.state_dict(), os.path.join(opt.checkpoints_dir, symbol_with_optimer.symbol.op_name + '_latest.pth') ) 

        # Test the result
        if epoch % 20 == 19:
            for symbol_with_optimer in symbol_list: 
                symbol_with_optimer.symbol.eval()
                loss_dic[symbol_with_optimer.symbol.op_name] = 0.0

            batch_cnt = 0
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    inputs, labels = data['img'], data['label']
                    inputs, labels = inputs.to(device), labels.to(device)

                    for symbol_with_optimer in symbol_list:
                        outputs, track_probe = symbol_with_optimer.symbol(inputs)
                        loss = criterion(outputs, labels)
                        loss_dic[symbol_with_optimer.symbol.op_name] += loss.item()
                    batch_cnt += 1

            result_info = 'Test Result------'
            for symbol_with_optimer in symbol_list:
                result_info += ' ' + symbol_with_optimer.symbol.op_name + ': %.5f' % math.sqrt(loss_dic[symbol_with_optimer.symbol.op_name] / batch_cnt)
            print result_info
            sys.stdout.flush()




        

