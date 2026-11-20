"""
# load ettin encoder tokenizer
name = "jhu-clsp/ettin-encoder-17m"
tokenizer = AutoTokenizer.from_pretrained(name)
"""

def create_batch_data(x,y,shuffle=True,batch=32):
    
    """
    create tensor batches
    x -> input text sequences
    y -> output label 
    
    for creating validation or test data, set `shuffle` as False
    and you the batch size can be twice than that of the training
    
    for eg. if for training, batch=32
    for testing or validation set, you can make it 64
    
    """
    
    x_ = tokenizer(x, return_tensors="pt",max_length=256,truncation=True,padding=True)
    y_ = torch.tensor(y)
    dataset = TensorDataset(x_["input_ids"], x_["attention_mask"], y_)
    data = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    return data
    


class CosineLRScheduler:
    def __init__(self, total_steps, peak_lr, min_lr, init_lr, warmup_rate):
        """
        Custom lr scheduler. linear lr warm-up and cosine lr decay.
        :param total_steps: total training steps
        :param max_lr: peak lr value
        :param min_lr: lr decay value
        :param init_lr: initial lr value
        :param warmup_rate: number of warmup steps
        """
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.total_setps = total_steps
        self.warmup = int(total_steps * warmup_rate)
        self.decay = total_steps - self.warmup
        self.warmup_lr = np.linspace(init_lr, peak_lr, self.warmup)
        self.max_lr = self.warmup_lr[-1]
        
        """
        usage example ->
        
        PEAK_LR = 5e-5
        END_LR = 1e-8
        INIT_LR = 1e-8
        WARMUP_RATE = 0.05
        EPOCHS = 5
        TOTAL_STEPS = len(train) * EPOCHS (total number of training or grad update steps)

        lr_scheduler = CosineLRScheduler(TOTAL_STEPS,
                                         peak_lr=PEAK_LR,
                                         min_lr=END_LR,
                                         init_lr=INIT_LR,
                                         warmup_rate=WARMUP_RATE)
                                         
                                    
        to view the lr scheduling curve using plt, run the following (after the previous step)
        
        plt.plot([lr_scheduler(i) for i in range(1,TOTAL_STEPS+1)])
        
        
        """


    def __call__(self, step):

        if step <= self.warmup:

            lr = self.warmup_lr[step - 1]
        else:
            step = step - self.warmup
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(step * np.pi / self.decay))

        return lr


class Classifier(nn.Module):
    
    def __init__(self,dim=256,n_classes=1,name="jhu-clsp/ettin-encoder-17m"):
        super().__init__()
        self.electra = AutoModel.from_pretrained(name)
        self.cls_head = nn.Linear(dim,n_classes)
        
    def masked_avg_pool(x,mask):
        if mask is None:
            return x.mean(dim=1)

        mask = (mask != 0).to(dtype=x.dtype).unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-8)  # Prevent division by zero
        return summed / counts
        
    def forward(self,x):
        ids,mask = x
        x = self.electra(input_ids=ids,attention_mask=mask)
        x = x['last_hidden_state']
        x = self.masked_avg_pool(x,mask)
        x = self.cls_head(x)
        return x
    
    
def train_model(data,model,EPOCHS,file='saved_weights.pt'):
    
    global losses
    
    opt = torch.optim.AdamW(model.parameters(),weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    train,test = data

    losses = {'train':[],'valid':[]}
    
    step = 1

    for e in range(1,EPOCHS + 1):

        print(f"{e}/{EPOCHS}")
        print(f"Learning Rate = {lr_scheduler(step)}")

        loss = 0
        model.train()

        for i,batch in enumerate(tqdm(train),1):

            lr = lr_scheduler(step)
            opt.param_groups[0]['lr'] = lr

            opt.zero_grad()

            ids,mask,y = batch
            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)
            y = y.long()
            pred = model([ids,mask])
            batch_loss = loss_fn(pred,y)
            batch_loss.backward()
            opt.step()

            loss += batch_loss.item()
            
            step += 1

        loss = round(loss / len(train),4)
        losses['train'].append(loss)
        train_loss = loss

        model.eval()
        loss = 0

        with torch.no_grad():

            for i,batch in enumerate(tqdm(test)):

                ids,mask,y = batch
                ids = ids.to(device)
                mask = mask.to(device)
                y = y.to(device)
                y = y.long() 
                pred = model([ids,mask])
                batch_loss = loss_fn(pred,y)
                loss += batch_loss.item()
        
        print(f'Train Loss = {train_loss}')
        loss = round(loss / len(test),4)
        print(f"Val Loss = {loss}")


        if e == 1 or loss < min(losses['valid']):
            torch.save(model.state_dict(),file)
            print("--weights saved")

        losses['valid'].append(loss)
        print('--------------------------------------')
        print()
