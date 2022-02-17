import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import gdl

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, pool=None):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.pool = pool

        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(
                    gdl.SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx], self.pool))
            else:
                self.en_layers.append(
                    gdl.SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx], self.pool))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(
            nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(
                    gdl.SpiralDeblock(out_channels[-idx - 1],
                                  out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    gdl.SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
                                  self.spiral_indices[-idx - 1]))
        self.de_layers.append(
            gdl.SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def forward(self, x, *indices):
        z = self.encoder(x)
        # out = self.decoder(z)
        return z

## net for triplet loss plus softmax
class Encoder_sf(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, pool=None):
        super(Encoder_sf, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.num_vert = self.down_transform[-1].size(0)
        self.pool = pool
        self.elu = nn.ELU()
        # encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)-1):
            if idx == 0:
                self.en_layers.append(
                    gdl.SpiralEnblock(in_channels, out_channels[idx],
                                  self.spiral_indices[idx], self.pool))
            else:
                self.en_layers.append(
                    gdl.SpiralEnblock(out_channels[idx - 1], out_channels[idx],
                                  self.spiral_indices[idx], self.pool))
        self.en_layers.append(
            nn.Linear(self.num_vert * out_channels[-2], out_channels[-1]))
        self.en_layers.append(
            nn.Linear( out_channels[-1], latent_channels))

        # decoder
        # self.de_layers = nn.ModuleList()
        # self.de_layers.append(
        #     nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        # for idx in range(len(out_channels)):
        #     if idx == 0:
        #         self.de_layers.append(
        #             gdl.SpiralDeblock(out_channels[-idx - 1],
        #                           out_channels[-idx - 1],
        #                           self.spiral_indices[-idx - 1]))
        #     else:
        #         self.de_layers.append(
        #             gdl.SpiralDeblock(out_channels[-idx], out_channels[-idx - 1],
        #                           self.spiral_indices[-idx - 1]))
        # self.de_layers.append(
        #     gdl.SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i < len(self.en_layers) - 2:
                x = layer(x, self.down_transform[i])
            elif i == len(self.en_layers)-2:
                x = x.view(-1, layer.weight.size(1))
                x = self.elu(layer(x))
            elif i == len(self.en_layers)-1:
                y= layer(x)
        return x,y

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def forward(self, x, *indices):
        x,y = self.encoder(x)
        # out = self.decoder(z)
        return x,y

# triplet net for softmax plus triplet loss
class TripletNet_Softmax(nn.Module):
    def __init__(self, model, triplet_selector,args):
        super(TripletNet_Softmax, self).__init__()
        self.model = model
        self.triplet_selector = triplet_selector
        self.test_mode = args.test  # to be changed and added to the argument



    def forward(self, faces, labels):
        faces=torch.squeeze(faces)
        if (self.test_mode == True):
            embedded_anchors,output = self.model(faces)
            embedded_positives = embedded_anchors
            embedded_negatives = embedded_anchors
        else:
            embeddings,output = self.model(faces)
            # print("*************labels.shape = ", labels.shape)
            # print("*************embeddings.shape = ", embeddings.shape)
            # print("*************triplet selector = ",self.triplet_selector)
            with torch.no_grad():# Calculation of the selection of the triplets is of no use for the backpropagation
                anchor_idxs, pos_idxs, neg_idxs = self.triplet_selector.get_triplets(embeddings=embeddings, labels=labels)

            embedded_anchors = embeddings[anchor_idxs]
            embedded_positives = embeddings[pos_idxs]
            embedded_negatives = embeddings[neg_idxs]

        # Note that the distances might already be calculated (by for example the BatchHard TripletSelector), however
        # using these distances proved to be numerically unstable when applying the backpropagation.

        pos_dists = F.pairwise_distance(embedded_anchors, embedded_positives, 2)
        neg_dists = F.pairwise_distance(embedded_anchors, embedded_negatives, 2)

        # pos_dists = 2-F.cosine_similarity(embedded_anchors, embedded_positives, 1)+1
        # neg_dists = 2-F.cosine_similarity(embedded_anchors, embedded_negatives, 1)+1
        if (self.test_mode == True):
            return pos_dists, neg_dists, output, embedded_anchors
        else:
            return pos_dists, neg_dists, output


class TripletNet(nn.Module):
    def __init__(self, model, triplet_selector,args):
        super(TripletNet, self).__init__()
        self.model = model
        self.triplet_selector = triplet_selector
        self.test_mode = args.test  # to be changed and added to the argument



    def forward(self, faces, labels):
        faces=torch.squeeze(faces)
        if (self.test_mode == True):
            embeddings = self.model(faces)
            embedded_positives = embeddings
            embedded_negatives = embeddings
            embedded_anchors = embeddings
        else:
            embeddings = self.model(faces)
            # print("*************labels.shape = ", labels.shape)
            # print("*************embeddings.shape = ", embeddings.shape)
            # print("*************triplet selector = ",self.triplet_selector)
            with torch.no_grad():# Calculation of the selection of the triplets is of no use for the backpropagation
                anchor_idxs, pos_idxs, neg_idxs = self.triplet_selector.get_triplets(embeddings=embeddings, labels=labels)

            embedded_anchors = embeddings[anchor_idxs]
            embedded_positives = embeddings[pos_idxs]
            embedded_negatives = embeddings[neg_idxs]

        # Note that the distances might already be calculated (by for example the BatchHard TripletSelector), however
        # using these distances proved to be numerically unstable when applying the backpropagation.

        pos_dists = F.pairwise_distance(embedded_anchors, embedded_positives, 2)
        neg_dists = F.pairwise_distance(embedded_anchors, embedded_negatives, 2)

        # pos_dists = 2-F.cosine_similarity(embedded_anchors, embedded_positives, 1)+1
        # neg_dists = 2-F.cosine_similarity(embedded_anchors, embedded_negatives, 1)+1

        return pos_dists, neg_dists, embeddings
