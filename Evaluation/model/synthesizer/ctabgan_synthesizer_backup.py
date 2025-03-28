import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from model.privacy_utils.rdp_accountant import compute_rdp, get_privacy_spent
from tqdm import tqdm
import csv 


class Classifier(Module):
    def __init__(self,input_dim, dis_dims,st_ed):
        super(Classifier,self).__init__()
        
        dim = input_dim-(st_ed[1]-st_ed[0])
        seq = []
        self.str_end = st_ed
        
        for item in list(dis_dims):
            
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            
            dim = item
        
        if (st_ed[1]-st_ed[0])==1:
            seq += [Linear(dim, 1)]
        
        elif (st_ed[1]-st_ed[0])==2:
            seq += [Linear(dim, 1),Sigmoid()]
        else:
            seq += [Linear(dim,(st_ed[1]-st_ed[0]))] 
        
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        label=None
        
        if (self.str_end[1]-self.str_end[0])==1:
            label = input[:, self.str_end[0]:self.str_end[1]]
        else:
            label = input[:, self.str_end[0]:self.str_end[1]] 
        
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1)
        
        if ((self.str_end[1]-self.str_end[0])==2) | ((self.str_end[1]-self.str_end[0])==1):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label

def apply_activate(data, output_info):
    
    data_t = []
    st = 0
    
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    
    return torch.cat(data_t, dim=1)

def get_st_ed(target_col_index,problem_type,output_info):
    
    st = 0
    c= 0
    tc= 0

    for item in output_info:
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
            if item[2] == 'yes_g':
                c+=1
        elif item[1] == 'softmax':
            st += item[0]
            c+=1
        tc+=1    
    
    if problem_type == "Classification":
        ed= st+output_info[tc][0] 
    else:
        if output_info[tc][2] == "yes_g":
            ed = st+output_info[tc][0]
        else: ed = st + (1+output_info[tc+1][0])

    return (st,ed)

def random_choice_prob_index_sampling(probs,col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        pp = probs[i] + 1e-6
        pp = pp / sum(pp)
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index_training(probs,col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        pp = probs[i] + 1e-6
        pp = pp / sum(pp)
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Cond(object):
    def __init__(self, data, output_info):
       
        self.model = []
        st = 0
        for item in output_info:
           
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            
        self.interval = []
        self.n_col = 0  
        self.n_opt = 0  
        st = 0
        self.p_training = []
        self.p_sampling = []
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax': 
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)  
                tmp_sampling = np.sum(data[:, st:ed], axis=0)     
                tmp = np.log(tmp + 1)  
                tmp = tmp / np.sum(tmp) 
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p_training.append(tmp) 
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
                
        self.interval = np.asarray(self.interval)
        
    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1  
        opt1prime = random_choice_prob_index_training(self.p_training,idx)
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
      
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,idx)
        
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec
    
    def sample_minority(self, batch):
        
        if self.n_col == 0:
            return None
        
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.ones(batch, dtype = int) * (self.n_col-1)
       
        for i in range(batch):
            col = idx[i]   
            pick = int(1)
            vec[i, pick + self.interval[col, 0]] = 1
            
        return vec 

def cond_loss(data, output_info, c, m):
    
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            continue

        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]

class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed
                
    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]

class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        info = len(layers)-2
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, input):
        return (self.seq(input)), self.seq_info(input)

class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)

def determine_layers_disc(side, num_channels):
    assert side >= 4 and side <= 64

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []
    num_c = num_channels
    num_s = side / 2
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c * 2
        num_s = num_s / 2

    layers_D = []

    for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            LayerNorm(ln),
            LeakyReLU(0.2, inplace=True),
        ]

    layers_D += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)] 

    return layers_D

def determine_layers_gen(side, random_dim, num_channels):
    assert side >= 4 and side <= 64

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layerNorms = []

    num_c = num_channels * (2 ** (len(layer_dims) - 2))
    num_s = int(side / (2 ** (len(layer_dims) - 1)))
    for l in range(len(layer_dims) - 1):
        layerNorms.append([int(num_c), int(num_s), int(num_s)])
        num_c = num_c / 2
        num_s = num_s * 2

    layers_G = [ConvTranspose2d(random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)]

    for prev, curr, ln in zip(reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms):
        layers_G += [LayerNorm(ln), ReLU(True), ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)]
    return layers_G

def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so)*low + (torch.sin(val*omega)/so) * high
    
    return res

def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer, device='cpu', lambda_=10):
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1,  device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    interpolates = transformer.transform(interpolates)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates,_ = netD(interpolates) 

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0] 
    
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_
    
    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)

class CTABGANSynthesizer:
    def __init__(self,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=300):
                 
        self.private = False
        self.micro_batch_size = 64
        self.clip_coeff = 1
        self.sigma = 1.02
        self.target_delta = 1e-5

        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, general=[], non_categorical=[], type={}):
        print("init for all the loss record")
        d_real_list = []
        d_fake_list = []
        pen_list = []
        cross_list = []
        g_list = []
        info_list = []
        cc_list = []
        cg_list = []

        problem_type = None
        target_index=None
        if type:
            problem_type = list(type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(type[problem_type])

        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed, general_list=general, non_categorical_list=non_categorical)
        self.transformer.fit() 
        train_data = self.transformer.transform(train_data.values)
        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)
        		
        sides = [4, 8, 16, 24, 32, 64]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break
        
        sides = [4, 8, 16, 24, 32, 64]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break
		

        layers_G = determine_layers_gen(self.gside, self.random_dim+self.cond_generator.n_opt, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        
        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)
        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier=None
        optimizerC= None
        if target_index != None:
            st_ed= get_st_ed(target_index,problem_type,self.transformer.output_info)
            classifier = Classifier(data_dim,self.class_dim,st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(),**optimizer_params)
        
        
        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)       
        self.Dtransformer = ImageTransformer(self.dside)
        
        epsilon = 0
        epoch = 0
        steps = 0
        ci = 1
        
        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in tqdm(range(self.epochs)):
            for id_ in range(steps_per_epoch):
				
                
                for _ in range(ci):
                    noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                    condvec = self.cond_generator.sample_train(self.batch_size)

                    c, m, col, opt = condvec
                    c = torch.from_numpy(c).to(self.device)
                    m = torch.from_numpy(m).to(self.device)
                    noisez = torch.cat([noisez, c], dim=1)
                    noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                    
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = c[perm]
                    
                    real = torch.from_numpy(real.astype('float32')).to(self.device)
                    
                    fake = self.generator(noisez)
                    faket = self.Gtransformer.inverse_transform(fake)
                    fakeact = apply_activate(faket, self.transformer.output_info)
                    
                    fake_cat = torch.cat([fakeact, c], dim=1)
                    real_cat = torch.cat([real, c_perm], dim=1)
                    
                    real_cat_d = self.Dtransformer.transform(real_cat)
                    fake_cat_d = self.Dtransformer.transform(fake_cat)
                    
                    optimizerD.zero_grad()
                    
                    d_real,_ = discriminator(real_cat_d)
                    
                    
                    if self.private:
                        
                        clipped_grads = {
                            name: torch.zeros_like(param) for name, param in discriminator.named_parameters()}

                        for k in range(int(d_real.size(0) / self.micro_batch_size)):
                            err_micro = -1*d_real[k * self.micro_batch_size: (k + 1) * self.micro_batch_size].mean(0).view(1)
                            err_micro.backward(retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), self.clip_coeff)
                            for name, param in discriminator.named_parameters():
                                clipped_grads[name] += param.grad
                            discriminator.zero_grad()

                        for name, param in discriminator.named_parameters():
                            param.grad = (clipped_grads[name] + torch.FloatTensor(
                                clipped_grads[name].size()).normal_(0, self.sigma * self.clip_coeff).cuda()) / (
                                                     d_real.size(0) / self.micro_batch_size)

                        steps += 1
                
                    else: 
                        d_real = -torch.mean(d_real)
                        d_real_list.append(d_real.item())
                        d_real.backward() 
                    

                    d_fake,_ = discriminator(fake_cat_d)
                    
                    d_fake = torch.mean(d_fake)
                    d_fake_list.append(d_fake.item())
                    d_fake.backward() 
                    
                    pen = calc_gradient_penalty_slerp(discriminator, real_cat, fake_cat,  self.Dtransformer , self.device)
                    pen_list.append(pen.item())
                    pen.backward()
                
                    optimizerD.step()
                    
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                
                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)

                optimizerG.zero_grad()

                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1) 
                fake_cat = self.Dtransformer.transform(fake_cat)
                    
                y_fake,info_fake = discriminator(fake_cat)
                
                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)
                cross_list.append(cross_entropy.item())
                _,info_real = discriminator(real_cat_d)
                

                g = -torch.mean(y_fake) + cross_entropy
                g_list.append(g.item())
                g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size,-1), dim=0) - torch.mean(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size,-1), dim=0) - torch.std(info_real.view(self.batch_size,-1), dim=0), 1)
                loss_info = loss_mean + loss_std 
                info_list.append(loss_info.item())
                loss_info.backward()
                optimizerG.step()


                if problem_type:
                    
                    if problem_type == "Classification":
                    
                        c_loss = None 
                        if (st_ed[1] - st_ed[0])==2:
                            c_loss = BCELoss()
                        else: c_loss = CrossEntropyLoss()   
                        
                        optimizerC.zero_grad()
                        real_pre, real_label = classifier(real)
                        real_label = torch.argmax(real_label, axis=-1)
                        if (st_ed[1] - st_ed[0])==2:
                            real_label = real_label.type_as(real_pre)
                        loss_cc = c_loss(real_pre, real_label)
                        cc_list.append(loss_cc.item())
                        loss_cc.backward()
                        optimizerC.step()
                    
                        optimizerG.zero_grad()
                        fake = self.generator(noisez)
                        faket = self.Gtransformer.inverse_transform(fake)
                        fakeact = apply_activate(faket, self.transformer.output_info)
                        fake_pre, fake_label = classifier(fakeact)        
                        fake_label = torch.argmax(fake_label, axis=-1)
                        if (st_ed[1] - st_ed[0])==2:
                            fake_label = fake_label.type_as(fake_pre)
                        loss_cg = c_loss(fake_pre, fake_label)
                        cg_list.append(loss_cg.item())
                        loss_cg.backward()
                        optimizerG.step()
                    
                    else:
                        r_loss = None
                        if (st_ed[1]-st_ed[0]) == 1:
                            r_loss = SmoothL1Loss()
                            optimizerC.zero_grad()
                            real_pre, real_label = classifier(real)
                            real_label = real_label.type_as(real_pre)
                            real_label = torch.reshape(real_label,real_pre.size())
                            loss_cc = r_loss(real_pre, real_label)
                            loss_cc.backward()
                            optimizerC.step()
                        
                            optimizerG.zero_grad()
                            fake = self.generator(noisez)
                            faket = self.Gtransformer.inverse_transform(fake)
                            fakeact = apply_activate(faket, self.transformer.output_info)
                            fake_pre, fake_label = classifier(fakeact)        
                            fake_label = fake_label.type_as(fake_pre)
                            fake_label = torch.reshape(fake_label,fake_pre.size())
                            loss_cg = r_loss(fake_pre, fake_label)
                            loss_cg.backward()
                            optimizerG.step()
                            
                        else: 
                            r_loss = (SmoothL1Loss(),CrossEntropyLoss())
                            optimizerC.zero_grad()
                            real_pre, real_label = classifier(real)
                            real_pre_r, real_label_r = real_pre[:,0], real_label[:,0]
                            real_label_r = real_label_r.type_as(real_pre_r)
                            real_label_r = torch.reshape(real_label_r,real_pre_r.size())
                            real_pre_c, real_label_c =  real_pre[:,1:], real_label[:,1:]
                            real_label_c = torch.argmax(real_label_c, axis=-1)
                            loss_cc = r_loss[0](real_pre_r, real_label_r) + r_loss[1](real_pre_c, real_label_c)  
                            loss_cc.backward()
                            optimizerC.step()
                        
                            optimizerG.zero_grad()
                            fake = self.generator(noisez)
                            faket = self.Gtransformer.inverse_transform(fake)
                            fakeact = apply_activate(faket, self.transformer.output_info)
                            fake_pre, fake_label = classifier(fakeact)
                            fake_pre_r, fake_label_r = fake_pre[:,0], fake_label[:,0]
                            fake_label_r = fake_label_r.type_as(fake_pre_r)
                            fake_label_r = torch.reshape(fake_label_r,fake_pre_r.size())
                            fake_pre_c, fake_label_c =  fake_pre[:,1:], fake_label[:,1:]
                            fake_label_c = torch.argmax(fake_label_c, axis=-1)
                            loss_cg = r_loss[0](fake_pre_r, fake_label_r) + r_loss[1](fake_pre_c, fake_label_c)              
                            loss_cg.backward()
                            optimizerG.step()


            epoch += 1
            if self.private:
                max_lmbd = 4095
                lmbds = range(2, max_lmbd + 1)
                rdp = compute_rdp(self.batch_size / train_data.shape[0], self.sigma, steps, lmbds)
                epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5)
                print("Epoch :", epoch, "Epsilon spent : ", epsilon)

        print("d_real_list: ", d_real_list)  
        print("d_fake_list: ", d_fake_list)  
        print("pen_list: ", pen_list)  
        print("cross_list: ", cross_list)  
        print("g_list: ", g_list)  
        print("info_list: ", info_list)  
        print("cc_list: ", cc_list)  
        print("cg_list: ", cg_list)  
        with open('d_real_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(d_real_list)   
        with open('d_fake_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(d_fake_list)  
        with open('pen_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(pen_list)  
        with open('cross_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(cross_list)  
        with open('g_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(g_list)  
        with open('info_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(info_list)  
        with open('cc_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(cc_list)  
        with open('cg_list.csv', 'w') as f: 
            write = csv.writer(f) 
            write.writerow(cg_list)  
















    def sample(self, n):
        
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        
        data = []
        
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket,output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result,resample = self.transformer.inverse_transform(data)
        
        while len(result) < n:
            data_resample = []    
            steps_left = resample// self.batch_size + 1
            
            for i in range(steps_left):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(self.batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                    
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            data_resample = np.concatenate(data_resample, axis=0)

            res,resample = self.transformer.inverse_transform(data_resample)
            result  = np.concatenate([result,res],axis=0)
        
        return result[0:n]

    
    def sample_minority(self, n):
        
        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        
        data = []
        
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample_minority(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket,output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result,resample = self.transformer.inverse_transform(data)

        while len(result) < n:
            
            data_resample = []    
            steps_left = resample// self.batch_size + 1

            for i in range(steps_left):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample_minority(self.batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez =  noisez.view(self.batch_size,self.random_dim+self.cond_generator.n_opt,1,1)
                    
                fake = self.generator(noisez)
                faket = self.Gtransformer.inverse_transform(fake)
                fakeact = apply_activate(faket,output_info)
                data_resample.append(fakeact.detach().cpu().numpy())
            
            data_resample = np.concatenate(data_resample, axis=0)
            res,resample = self.transformer.inverse_transform(data_resample)
            result  = np.concatenate([result,res],axis=0)
        
        return result[0:n]

