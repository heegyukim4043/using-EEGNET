from pytorch_eegnet import EEGNet_torch
from pytorch_Shallow import ShallowConvNet
from pytorch_deepconvnet import DeepConvNet
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib as mpl
import scipy.io
import os
import torch
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from captum.attr import DeepLift
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = [0]
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device: ', device)

kernel_size, chans, samples = 1, 32, 512
result_path = './result/shallow/1000epoch_TGI'
#Shallow/test_500_cold_dropout_p5_triple' 
#torch_5fold_500epoch_cold_repeated' #'./result_512_bi_CH'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

exclude = [22,30]
'''
model = EEGNet_torch(nb_classes = 6, Chans = chans, Samples = samples, 
                dropoutRate=0.7, kernelLength=32, F1=8, D=2, F2=16,
                dropoutType='Dropout').to(device)
'''
model = ShallowConvNet(nb_classes =3, Chans = chans, Samples = samples, 
                        dropoutRate = 0.7).to(device)
'''
model = DeepConvNet(nb_classes =6, Chans = chans, Samples = samples, 
                        dropoutRate = 0.7).to(device)
'''
class_weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = opt.Adam(model.parameters())

# Train the model
for sub in [x for x in range(1, 44) if x not in exclude]:
    dat_name = (".\data_deep\labelsrand_5fold_TGI\subject%02d_traindata.mat" %(sub))
    seq_name = ('.\data_deep\labelsrand_5fold_TGI\subject%02d_trainseq.mat' %(sub))

    train_data_file = scipy.io.loadmat(dat_name)
    train_seq_file = scipy.io.loadmat(seq_name)

    test_name = (".\data_deep\labelsrand_5fold_TGI\subject%02d_testdata.mat" %(sub))
    test_seq_name = ('.\data_deep\labelsrand_5fold_TGI\subject%02d_testseq.mat' %(sub))

    test_data_file = scipy.io.loadmat(test_name)
    test_seq_file = scipy.io.loadmat(test_seq_name)

    total_acc = []
    
    i = [1, 2, 3, 4, 5]
    random.shuffle(i)


    for j in range(1,6):
        nums = i[j-1]
        data_train = train_data_file['train_EEG']
        data_train = data_train[0]
        data_train = data_train[nums - 1]
        
        seq_train = train_seq_file['train_seq']
        seq_train = seq_train[0]
        seq_train = seq_train[nums-1]
        
        data_test = test_data_file['test_EEG']
        data_test = data_test[0]       
        data_test = data_test[nums - 1]
        
        seq_test = test_seq_file['test_seq']
        seq_test = seq_test[0]
        seq_test = seq_test[nums-1]
        '''
        checkpointer = ModelCheckpoint(filepath='./tmp/checkpoint_s' + str(sub) + '_' + str(nums) +
                                '.h5', verbose = 0, save_best_only = True)
                                '''                    
        #data_train = torch.from_numpy(data_train)
        data_train =torch.from_numpy(data_train.transpose(2,0,1)).float().unsqueeze(-1)
        print(np.shape(data_train))        
        seq_train =  torch.from_numpy(seq_train).squeeze().long()

        print(np.shape(seq_train))

        data_test = torch.from_numpy(data_test.transpose(2,0,1)).float().unsqueeze(-1)
        print(np.shape(data_test))
        seq_test =  torch.from_numpy(seq_test).squeeze().long()
        
        data_train, data_val, train_label, seq_val = train_test_split(
            data_train, seq_train, test_size = 0.2, stratify = seq_train)
        print(np.shape(train_label), 'size')

        num_classes = torch.unique(train_label).size(0)
        print(num_classes)
        
        train_label = F.one_hot(train_label.long() - 1)
        seq_val = F.one_hot(seq_val.long() - 1)
        seq_test = F.one_hot(seq_test.long() - 1)

        batch_size = 32
        print(np.shape(train_label), 'seq_train')
        print(np.shape(data_train))
        train_dataset = TensorDataset(data_train.permute(0,3,1,2).to(device), 
                                      train_label.to(device))
        print(np.shape(data_train))
        train_loader = DataLoader(train_dataset, batch_size)
        val_dataset = TensorDataset(data_val.permute(0,3,1,2).to(device), 
                                    seq_val.to(device))
        val_loader = DataLoader(val_dataset, batch_size)
        test_dataset = TensorDataset(data_test.permute(0,3,1,2).to(device), 
                                     seq_test.to(device))
        test_loader = DataLoader(test_dataset)
       # print(train_loader, 'input dataset')
     
        train_acc = []
        val_acc = []           
        for epoch in range(1000):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                epoch_loss = running_loss / inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets.argmax(dim=-1)).sum().item()
            train_acc.append(correct/total)
            print('Epoch %d training loss: %.3f' % (epoch+1, loss.item()))
            
            
            model.eval()
            val_correct = 0
            val_total = 0
            for inputs, targets in val_loader:
                val_inputs = inputs
                val_outputs = model(val_inputs)
                val_targets = targets.float()
                val_loss = criterion(val_outputs, val_targets)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (val_predicted == targets.argmax(dim=-1)).sum().item()
            val_acc.append(val_correct/val_total)
            print('Epoch %d validation loss: %.3f' % (epoch+1, val_loss.item()))
    
    
    
        # Save the PyTorch model
        torch.save(model.state_dict(), result_path +'/weight_final_sub' + 
                   str(sub) + '.pth')

        # Load the PyTorch model
        # model.load_state_dict(torch.load('./tmp/weight_final_sub' + str(sub) + '.pth'))

        # Make prediction on test set
        with torch.no_grad():
            model.eval()
            probs = F.softmax(model(data_test.permute(0,3,1,2).to(device)), dim=-1)
            preds = torch.argmax(probs, dim=-1)
            targets = targets.float()
            acc = torch.mean((preds == seq_test.argmax(dim=-1).to(device)).float())
            total_acc.append(acc)
            print("Classification accuracy: ", total_acc)
            preds = preds.cpu().numpy()
            seq_test = seq_test.argmax(dim=-1).cpu().numpy()
            deep_lift = DeepLift(model)
            labels = train_label.argmax(dim=-1).to(device)
           # print(labels)
            attr_0 = deep_lift.attribute(data_train.permute(0,3,1,2).to(device), 
                                         target = labels)
           # print(np.shape(attr_0))
            attr_0 = attr_0.cpu().numpy()          
            scipy.io.savemat(result_path + '/Subject' + str(sub)+'_'+str(nums) + 
                             '.mat',
                        {'test_classified_Label':preds,
                         'test_true_label': seq_test,
                         'attributions': attr_0,
                         'labels':train_label.argmax(dim=-1).cpu().numpy(),
                         'train_acc': train_acc,
                         'val_acc': val_acc
                         })
        
    total_acc = np.array([tensor.cpu().numpy() for tensor in total_acc])
    mat_result_file = ('/result_label_Subject%02d_5fold.mat' %(sub))
    scipy.io.savemat(result_path + mat_result_file,
                {'acc': total_acc, 'seq': i}) #, 'seq': i
    
           

