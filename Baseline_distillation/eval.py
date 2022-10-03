import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn

@torch.no_grad()
def eval(model, dataset, device, loss_func=nn.CrossEntropyLoss()):
    print('Model Evaluating')
    model = model.to(device)
    model.eval()

    batch_loss, batch_acc = 0, 0
    pbar = enumerate(tqdm(dataset, total=len(dataset)))
    start_time = time.time()

    for batch, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels)
        output_index = torch.argmax(outputs, dim=1)
        acc = (output_index==labels).sum()/len(outputs)

        batch_loss += loss.item()
        batch_acc += acc.item()

    end_time = time.time()

    print(f'\nTotal time for testing is {end_time-start_time:.2f}s')
    print(f'\nAverage loss: {batch_loss/(batch+1):.3f}  accuracy: {batch_acc/(batch+1):.3f}')