import time
import os
import torch
from tqdm import tqdm
import copy
import scipy.io as sio
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def run(model, train_loader, val_loader, loss_fn, epochs, optimizer, scheduler,
        writer, viz, device, start_epoch=1):
    for epoch in range(start_epoch, epochs + 1):
        t = time.time()
        # train_loss, train_acc = train(model, optimizer, train_loader, loss_fn, device)
        train_loss = train(model, optimizer, train_loader, loss_fn, device)
        t_duration = time.time() - t
        # val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        info = {
            'current_epoch': epoch,
            'epochs': epochs,
            'train_loss': train_loss,
            'validation_loss': val_loss,
            't_duration': t_duration
        }

        writer.print_info(info)
        writer.save_checkpoint(model, optimizer, scheduler, epoch, 0)
        if epoch % 50 == 0:
            writer.save_checkpoint(model, optimizer, scheduler, epoch, epoch)
        viz.line(torch.ones((1,)).cpu() * epoch, torch.Tensor([train_loss, val_loss]).unsqueeze(0).cpu(),
                 'Loss', ['Training loss', 'Validation loss'])
        # viz.line(torch.ones((1,)).cpu() * epoch, torch.Tensor([train_acc, val_acc]).unsqueeze(0).cpu(),
        #          'Accuracy', ['Training acc', 'Validation acc'])
        viz.save()


def train(model, optimizer, loader, loss_fn, device):
    model.train()

    total_loss = 0
    for data,grp in tqdm(loader):
        optimizer.zero_grad()
        x = data.to(device)
        grp=grp.long().to(device)
        pos_dist, neg_dist, emb_anc = model(x,grp)
        target = torch.ones(pos_dist.size()).cuda()
        loss = loss_fn(neg_dist, pos_dist, target)
        # acc=accuracy(out=out,target=grp)
        # print("******",loss.item())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    return (total_loss / len(loader))#,acc


def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for data,grp in loader:
            x = data.to(device)
            grp = grp.long().to(device)
            pos_dist, neg_dist, emb_anc = model(x,grp)
            target = torch.ones(pos_dist.size()).cuda()
            total_loss += loss_fn(neg_dist, pos_dist, target)
            # acc=accuracy(out=pred,target=grp)
    return (total_loss / len(loader))#,acc

def accuracy(out,target):

    probs = torch.softmax(out, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == target)
    accuracy = corrects.sum().float() / float(target.size(0))
    # accuracy = (torch.softmax(out, dim=1).argmax(dim=1) == target).sum().float() / float(target.size(0))
    return accuracy

def conf_mat(y_true, y_pred):
    c_mat= confusion_matrix(y_true, y_pred)
    return c_mat


def eval_error_tripletloss(model, loader, device, out_dir, results_dir, epoch, viz=None,test_val=False,test_controls=False, fold=0,plot_emb=False):
    model.eval()
    n = loader.dataset.get_total_size()

    with torch.no_grad():
        for b, [data,grp] in enumerate(loader):
            x = data.to(device)
            grp=grp.long().to(device)
            pos_dist, neg_dist, emb_anc = model(x,grp)
            # emb = model.encoder(x)
            # print("*** ",prediction.shape)

            if b == 0:
                original = copy.deepcopy(x)
                predictions = copy.deepcopy(emb_anc)
                # embedding = copy.deepcopy(emb)
                grps=copy.deepcopy(grp)
            else:
                original = torch.cat([original, x], 0)
                predictions = torch.cat([predictions, emb_anc], 0)
                # embedding = torch.cat([embedding, emb], 0)
                grps=torch.cat([grps,grp],0)
        print("*** ", predictions.shape, original.shape, grps.shape)
        # loss = loss_fn(predictions, grps)
        # acc=accuracy(out=predictions,target=grps)
        # print("**** accuracy is = ",acc)

    # probs = torch.softmax(predictions, dim=1)
    original = original.cpu().numpy()
    predictions = predictions.cpu().numpy()
    # probs=probs.cpu().numpy()
    # print(results_dir)
    if (test_controls == True):
        sio.savemat(os.path.join(results_dir, '_batch' + str(fold) + '_test_RPCP_predictions_{0}.mat'.format(epoch)),
                    {'original': original,
                     'predicted': predictions, 'grps': grps.cpu().numpy()})
    elif(test_val==False):
        sio.savemat(os.path.join(results_dir,'_batch'+str(fold)+ '_test_predictions_{0}.mat'.format(epoch)), {'original': original,
                                                                   'predicted': predictions,'grps':grps.cpu().numpy()})
    else:
        sio.savemat(os.path.join(results_dir, '_batch' + str(fold) + '_train_predictions_{0}.mat'.format(epoch)),
                    {'original': original,
                     'predicted': predictions, 'grps': grps.cpu().numpy()})

def eval_error_tripletloss_softmax(model, loader, device, out_dir, results_dir, epoch, viz=None, fold=0,plot_emb=False):
    model.eval()
    n = loader.dataset.get_total_size()

    with torch.no_grad():
        for b, [data,grp] in enumerate(loader):
            x = data.to(device)
            grp=grp.long().to(device)
            pos_dist, neg_dist, sf_output, emb = model(x,grp)
            # emb = model.encoder(x)
            # print("*** ",prediction.shape)

            if b == 0:
                original = copy.deepcopy(x)
                predictions = copy.deepcopy(sf_output)
                embedding = copy.deepcopy(emb)
                grps=copy.deepcopy(grp)
            else:
                original = torch.cat([original, x], 0)
                predictions = torch.cat([predictions, sf_output], 0)
                embedding = torch.cat([embedding, emb], 0)
                grps=torch.cat([grps,grp],0)
        print("*** ", predictions.shape, original.shape, grps.shape)
        # loss = loss_fn(predictions, grps)
        # acc=accuracy(out=predictions,target=grps)
        # print("**** accuracy is = ",acc)

    probs = torch.softmax(predictions, dim=1)
    original = original.cpu().numpy()
    predictions = predictions.cpu().numpy()
    embedding=embedding.cpu().numpy()
    probs=probs.cpu().numpy()
    # print(results_dir)
    sio.savemat(os.path.join(results_dir,'_batch'+str(fold)+ '_predictions_{0}.mat'.format(epoch)), {'original': original,
                                                               'predicted': predictions,'grps':grps.cpu().numpy(),'embedding': embedding,'probs':probs})

