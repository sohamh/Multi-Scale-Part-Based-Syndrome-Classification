import numpy as np
import torch
import torch.nn.functional as F
import scipy
import sklearn


# region Implemented TripletSelectors:
def get_selector(params):
    selectors = {'batch_random': BatchRandom, 'batch_hard': BatchHard,
                 'batch_all_hard': BatchAllHard, 'batch_all_random': BatchAllRandom}
    return selectors[params.ts]()


class BatchRandom:
    """
    Basic triplet selector that returns random triplets for (at most) each anchor from the provided batch.
    Input: A batch of labels (size B where B is the batch size)
    Output: Three tensor lists (size V<=B) containing the indices that form the triplets as follows:
            For each anchor a at position i in anchor_idxs and with label labels[a], the corresponding positive is
            p=pos_idxs[i] with label labels[p]==labels[a] and the corresponding negative is n=neg_idxs[i] with label
            labels[n]!=labels[a]. The triplet's indices (a,p,n) correspond to their position in the labels list.

            We thank Bram De Cooman for helping us with implementation of Triplet Selection
    """

    def __init__(self, pos_margin=None):
        self.pos_margin = pos_margin

    def get_triplets(self, embeddings, labels):
        pos_idxs = torch.zeros(labels.size(0), dtype=torch.long, device=labels.device)
        neg_idxs = torch.zeros(labels.size(0), dtype=torch.long, device=labels.device)

        pos_mask, neg_mask, valid_idxs = _getTripletMasks(labels, self.pos_margin)
        idxs = torch.arange(labels.size(0), device=labels.device)
        for i in range(valid_idxs.size(0)):
            a = valid_idxs[i]
            # For the current anchor, create a list of all possible positives and negatives
            positives = idxs[pos_mask[a, :]]
            negatives = idxs[neg_mask[a, :]]

            randPosIdx = np.random.randint(0, positives.size(0))
            randNegIdx = np.random.randint(0, negatives.size(0))

            # Out of all posible positives and negatives, select one randomly
            pos_idxs[a] = positives[randPosIdx]
            neg_idxs[a] = negatives[randNegIdx]

        # Only return valid triplets (those for which there exist both a positive AND negative)
        anchor_idxs = valid_idxs
        pos_idxs = pos_idxs[valid_idxs]
        neg_idxs = neg_idxs[valid_idxs]

        return anchor_idxs, pos_idxs, neg_idxs


class BatchHard:
    """ Triplet selector that returns the hardest triplet for each anchor from the batch. """

    def __init__(self, K, pos_margin=None):
        self.pos_margin = pos_margin

    def get_triplets(self, embeddings, labels):
        distances = _getDistances(embeddings)
        pos_mask, neg_mask, valid_idxs = _getTripletMasks(labels, self.pos_margin)
        pos_idxs = _getHardestPositives(distances, pos_mask)
        neg_idxs = _getHardestNegatives(distances, neg_mask)

        # Only keep valid triplets:
        pos_idxs = pos_idxs[valid_idxs]
        neg_idxs = neg_idxs[valid_idxs]
        anchor_idxs = valid_idxs

        return anchor_idxs, pos_idxs, neg_idxs


class BatchAll:
    """ Triplet selector that returns all possible triplets from the batch. """

    def __init__(self, K, pos_margin=None):
        self.pos_margin = pos_margin

    def get_triplets(self, embeddings, labels):
        pos_mask, neg_mask, _ = _getTripletMasks(labels, self.pos_margin)
        # Triplet mask is a 3D tensor, where element (a,p,n) is 1 only if the triplet with anchor a, positive p and
        # negative v is valid.
        triplet_mask = (pos_mask.unsqueeze(2) * neg_mask.unsqueeze(1)).to(torch.float32)
        # Retrieve anchor, positive and negative idxs of the valid triplets:
        triplet_idxs = torch.nonzero(triplet_mask)
        anchor_idxs = triplet_idxs[:, 0]
        pos_idxs = triplet_idxs[:, 1]
        neg_idxs = triplet_idxs[:, 2]

        return anchor_idxs, pos_idxs, neg_idxs


class BatchAllRandom(BatchAll):
    """ Triplet selector that returns (at most) K triplets out of all possible triplets from the batch. """

    def __init__(self, K, pos_margin=None):
        super().__init__(pos_margin)
        self.K = K

    def get_triplets(self, embeddings, labels):
        anchor_idxs, pos_idxs, neg_idxs = super().get_triplets(embeddings, labels)
        nb_triplets = int(anchor_idxs.size(0))

        triplet_idxs = np.random.choice(nb_triplets, min(nb_triplets, self.K), replace=False)

        return anchor_idxs[triplet_idxs], pos_idxs[triplet_idxs], neg_idxs[triplet_idxs]


class BatchAllHard(BatchAll):
    """ Triplet selector that returns (at most) the K hardest triplets from the batch. """

    def __init__(self, K, pos_margin=None):
        super().__init__(pos_margin)
        self.K = K

    def get_triplets(self, embeddings, labels):
        anchor_idxs, pos_idxs, neg_idxs = super().get_triplets(embeddings, labels)
        distances = _getDistances(embeddings)
        # Triplet losses is a 1D tensor, where each element i denotes the loss (without margin) of the triplet (a,p,n)
        # where a=anchor_idxs[i], p=pos_idxs[i] and n=neg_idxs[i].
        nb_triplets = int(anchor_idxs.size(0))
        triplet_losses = distances[anchor_idxs, pos_idxs] - distances[anchor_idxs, neg_idxs]
        # Select min(nb_triplets,K) hardest triplets:
        _, hardest_idxs = torch.topk(triplet_losses, min(nb_triplets, self.K))
        # Retrieve anchor, positive and negative indices of these hardest triplets:
        anchor_idxs = anchor_idxs[hardest_idxs]
        pos_idxs = pos_idxs[hardest_idxs]
        neg_idxs = neg_idxs[hardest_idxs]

        return anchor_idxs, pos_idxs, neg_idxs


# endregion

# region Utility functions used by the above TripletSelectors:

# Parts of this code are inspired by the tensorflow-triplet-loss implementation
def _getDistances(embeddings):
    """ Calculates all the pairwise distances between the different embeddings of the batch.
        As F.pdist only returns the upper-triangular part as a 1D-Tensor, numpy's triu and tril methods are
        used to create an easily accessible 2D distance Tensor where the i-j-th entry holds the pairwise
        distance between embeddings[i] and embeddings[j]. """
    # pdistances = F.pdist(embeddings, p=2)

    # pdistances = sklearn.metrics.pairwise.cosine_similarity(embeddings)+1
    # distances=torch.from_numpy(pdistances).float().to(embeddings.device)

    pdistances = scipy.spatial.distance.pdist(embeddings, 'minkowski', p=2)
    distances = torch.zeros((embeddings.size(0), embeddings.size(0)), device=embeddings.device)
    distances[np.triu_indices(embeddings.size(0), 1)] = torch.from_numpy(pdistances).float().to(embeddings.device)
    distances[np.tril_indices(embeddings.size(0), -1)] = torch.from_numpy(pdistances).float().to(embeddings.device)

    return distances


def _getHardestPositives(distances, pos_mask):
    """ For each anchor from the batch, determine the hardest positive by using the distances matrix and positives mask.
        The hardest positive corresponds to the furthest embedding that is marked as positive. """
    pos_distances = pos_mask.to(torch.float32) * distances  # * uses elementwise multiplication of tensors
    hard_indices = torch.argmax(pos_distances,
                                dim=1)  # Along each row, find the index with the largest positive distance

    return hard_indices


def _getHardestNegatives(distances, neg_mask):
    """ For each anchor from the batch, determine the hardest negative by using the distances matrix and negatives mask.
        The hardest negative corresponds to the closest embedding that is marked as negative. """
    same_label_mask = 1 - neg_mask

    neg_distances = neg_mask.to(torch.float32) * distances  # * uses elementwise multiplication of tensors
    max_neg_dist = torch.max(neg_distances)  # Extract maximum negative distance among the negative distances
    neg_distances = neg_distances + max_neg_dist * same_label_mask.to(
        torch.float32)  # And add it as a bias to the positive pairs (to prevent one of them from being the minimum)
    hard_indices = torch.argmin(neg_distances,
                                dim=1)  # Along each row, find the index with the smallest negative distance

    return hard_indices


def _getSameLabelMask(labels, pos_margin=None):
    """ From the given batch of labels, construct a mask of size B x B determining for each anchor which other embeddings
        are positive according to the given pos_margin. """
    # Input labels have size B x L where L is the size of 1 label. Unsqueezeing along dimension 1 creates a tensor of
    # size B x 1 x L and unsqueezing along dimension 0 creates a tensor of size 1 x B x L. Using torch.eq or taking the
    # difference applies pytorch's broadcoasting rules to these two tensors, resulting in a tensor of size B x B x L.
    # Finally, the torch.norm function takes the vector norm (2-norm) along the last dimension, thus resulting in a tensor
    # of size B x B x 1.
    if pos_margin is None:  # No margin means hard equality of the labels
        same_label_mask = torch.eq(torch.unsqueeze(labels, 1), torch.unsqueeze(labels, 0))
    elif len(pos_margin) == 1:  # Otherwise all labels withing pos_margin[0] are considered positive
        same_label_mask = torch.le(
            torch.norm(torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 0), dim=2, keepdim=True), pos_margin[0])
    else:  # In case pos_margin is multidimensional itself, the margin is applied for each label component separately
        # First convert pos_margin list to 1 x 1 x L tensor
        pos_margin = torch.unsqueeze(
            torch.unsqueeze(torch.tensor(pos_margin, device=labels.device, dtype=labels.dtype), 0), 0)
        # torch.le compares each label component separately and thus still returns a B x B x L tensor with ones and zeros
        # torch.prod is used to get a B x B x 1 mask with ones only if each component c is within its pos_margin[c]
        same_label_mask = torch.prod(
            torch.le(torch.abs(torch.unsqueeze(labels, 1) - torch.unsqueeze(labels, 0)), pos_margin), dim=2,
            keepdim=True)
        same_label_mask = same_label_mask.to(
            torch.uint8)  # Convert back to ByteTensor, as torch.prod returns a LongTensor
    return torch.squeeze(same_label_mask)  # Convert mask of size B x B x 1 to size B x B


def _getTripletMasks(labels, pos_margin=None):
    """ Constructs masks of size B x B denoting for each anchor which other embeddings can be used as positives and
        negatives for the creation of triplets. Each row i provides the mask for anchor i (i denoting the position in
        labels) and marks valid positives (or negatives) with 1 and invalid positives (or negatives) with 0. Finally, a
        valid_mask of size B is constructed marking which anchors can be used to create valid triplets. If valid_mask[i]
        equals 1, this means there is at least 1 positive and 1 negative that can be used to create a triplet. Otherwise
        valid_mask[i] equals 0. From this valid_mask a tensor with valid_idxs is constructed. """
    unequal_mask = 1 - torch.eye(labels.size(0), dtype=torch.uint8, device=labels.device)
    same_label_mask = _getSameLabelMask(labels, pos_margin)
    pos_mask = unequal_mask & same_label_mask  # Size B x B denoting for each embedding (in the role of anchor) which other embeddings are valid positives
    neg_mask = 1 - same_label_mask  # Size B x B denoting for each embedding (in the role of anchor) which other embeddings are valid negatives
    valid_mask = (torch.sum(pos_mask, 0) > 0) & (torch.sum(neg_mask,
                                                           0) > 0)  # Size B denoting for each embedding (in the role of anchor) whether a valid triplet can be created with the other embeddings
    idxs = torch.arange(labels.size(0), device=labels.device)
    return pos_mask, neg_mask, torch.masked_select(idxs, valid_mask)


# endregion
