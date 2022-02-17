from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io as sio


class DataLoaders:
    """ Creates mutual exclusive loaders for training, evaluation and testing backed by the same dataset. """

    def __init__(self, args, dataset):

        # tmp = sio.loadmat(args.sets_ind)
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

        if (args.train == True):
            all_idxs = list(range(dataset.get_total_size()))

            train_ind = all_idxs[:-args.nVal]
            val_ind = all_idxs[-args.nVal:]

            train_set = Subset(dataset, train_ind)
            val_set = Subset(dataset, val_ind)

            # self.weights=dataset.get_weights()

            # DataLoader used for training in batches:
            self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                                           **kwargs)  # collate_fun, num_workers
            # DataLoader used for evaluating:
            self.val_loader = DataLoader(val_set, batch_size=args.batch_size, **kwargs)
        if (args.test == True):

            test_set = dataset
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, **kwargs)




    def get_weights(self):
        return self.weights

