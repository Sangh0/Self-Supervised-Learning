import os
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

class TrainModel(object):
    
    def __init__(
        self,
        model,
        lr,
        device,
        optimizer,
        loss_func=None,
        phase='teacher',
        epochs=30,
        train_log_steps=300,
        valid_log_steps=100,
        check_point=False,
    ):
        assert phase in ('teacher', 'student'), \
            'you have to choose the phase between teacher and student'

        self.phase = phase

        self.device = device

        self.model = model.to(self.device)

        self.base_loss_func = nn.CrossEntropyLoss().to(self.device)
        self.loss_func = loss_func

        if optimizer=='adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
            )
        
        elif optimizer=='adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
            )
        
        elif optimizer=='sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
            )

        else:
            raise ValueError(f'The optimizer {optimizer} does not exist')

        self.epochs = epochs

        self.train_log_steps = train_log_steps
        self.valid_log_steps = valid_log_steps

        self.check_point = check_point

    def fit(self, train_data, validation_data):
        print('Start Model Training...!')
        start_training = time.time()
        min_val_loss = torch.inf
        loss_list, acc_list, val_loss_list, val_acc_list = [], [], [], []
        pbar = tqdm(range(self.epochs), total=int(self.epochs))
        for epoch in pbar:
            init_time = time.time()

            train_loss, train_acc = self.train_on_batch(
                train_data, self.train_log_steps,
            )

            loss_list.append(train_loss)
            acc_list.append(train_acc)

            valid_loss, valid_acc = self.validate_on_batch(
                validation_data, self.valid_log_steps,
            )

            val_loss_list.append(valid_loss)
            val_acc_list.append(valid_acc)

            end_time = time.time()

            print(f'\n{"="*40} Epoch {epoch+1}/{self.epochs} {"="*40}'
                  f'\ntime: {end_time-init_time:2f}s'
                  f'   lr = {self.optimizer.param_groups[0]["lr"]}')
            print(f'\ntrain average loss: {train_loss:.3f}'
                  f'   accuracy: {train_acc:.3f}')
            print(f'\nvalid average loss: {valid_loss:.3f}'
                  f'   accuracy: {valid_acc:.3f}')
            print(f'\n{"="*100}')

            if self.check_point:
                if valid_loss < min_val_loss:
                    print(f'Valid loss decreased ({min_val_loss:.3f}) --> ({valid_loss:.3f})  Saving model...')
                    os.makedirs('./weights', exist_ok=True)
                    torch.save(self.model.state_dict(), './weights/check_point_weights.pt')
                    min_val_loss = valid_loss

        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.2f}s')

        return {
            'model': self.model,
            'loss': loss_list,
            'acc_list': acc_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list,
        }

    @torch.no_grad()
    def validate_on_batch(self, validation_data, log_step):
        self.model.eval()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(validation_data):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.base_loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)

            batch_loss += loss.item()
            batch_acc += acc.item()

            if batch == 0:
                print(f'\n{" "*20} valid step {" "*20}')

            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(validation_data)}]'
                      f'  valid loss {loss:.3f}   accuracy: {acc:.3f}')

        return batch_loss/(batch+1), batch_acc/(batch+1)

    def train_on_batch(self, train_data, log_step):
        self.model.train()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(train_data):
            self.optimizer.zero_grad()

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            if self.phase == 'student':
                loss = self.loss_func(images, outputs, labels)
            else:
                loss = self.base_loss_func(outputs, labels)

            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)

            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()
            batch_acc += acc.item()

            if batch == 0:
                print(f'\n{" "*20} train step {" "*20}')

            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(train_data)}]'
                      f'  train loss: {loss:.3f}   accuracy: {acc:.3f}')

        return batch_loss/(batch+1), batch_acc/(batch+1)