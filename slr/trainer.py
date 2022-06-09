from PIL import Image

import torch
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

from .losses import focal_loss, water_obstacle_separation_loss, object_projection_loss, object_focal_loss
from .pairwise_affinity import pairwise_affinity_loss, object_pairwise_affinity_loss
from .metrics import PixelAccuracy, ClassIoU
from .utils import bool_arg

NUM_EPOCHS = {'warmup':25, 'finetune':50}
LEARNING_RATE = 1e-6
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6
LR_DECAY_POW = 0.9
FOCAL_LOSS_SCALE = 'labels'
SEPARATION_LOSS = {'warmup':False, 'finetune':True}
SL_LAMBDA = 0.01
PA_LOSS = {'warmup':True, 'finetune':True}
PA_LOSS_LAMBDA = 1
PA_LOSS_TAU = 0.1
OBJ_LOSS = {'warmup':True, 'finetune':False}
OBJ_LOSS_LAMBDA = 1

class LitModel(pl.LightningModule):
    """ Pytorch Lightning wrapper for a model, ready for distributed training. """

    @staticmethod
    def add_argparse_args(parser, phase):
        """Adds phase-specific model parameters to parser."""

        parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Momentum component of the optimiser.")
        parser.add_argument("--epochs", type=int, default=NUM_EPOCHS[phase],
                            help="Number of training epochs.")
        parser.add_argument("--lr-decay-pow", type=float, default=LR_DECAY_POW,
                            help="Decay parameter to compute the learning rate decay.")
        parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        parser.add_argument("--focal-loss-scale", type=str, default=FOCAL_LOSS_SCALE, choices=['logits', 'labels'],
                            help="Which scale to use for focal loss computation (logits or labels).")
        parser.add_argument("--separation-loss", default=SEPARATION_LOSS[phase], type=bool_arg,
                            help="Use separation loss.")
        parser.add_argument("--separation-loss-lambda", default=SL_LAMBDA, type=float,
                            help="The separation loss lambda (weight).")
        parser.add_argument("--separation-loss-sky", action='store_true',
                            help="Include sky in the separation loss computation.")
        parser.add_argument("--pairwise-affinity-loss", type=bool_arg, default=PA_LOSS[phase],
                            help="Add the pairwise affinity loss term.")
        parser.add_argument("--pairwise-affinity-loss-lambda", default=PA_LOSS_LAMBDA, type=float,
                            help="The pairwise affinity loss lambda (weight).")
        parser.add_argument("--pairwise-affinity-loss-tau", default=PA_LOSS_TAU, type=float,
                            help="The pairwise affinity loss tau (similarity threshold).")
        parser.add_argument("--object-loss", type=bool_arg, default=OBJ_LOSS[phase],
                            help="Add the object loss term.")
        parser.add_argument("--object-loss-pa", type=bool_arg, default=True,
                            help="Enable object loss PA term.")
        parser.add_argument("--object-loss-proj", type=bool_arg, default=True,
                            help="Enable object loss projection loss term.")
        parser.add_argument("--object-loss-aux", type=bool_arg, default=True,
                            help="Enable object loss auxiliary segmentation term.")
        parser.add_argument("--object-loss-lambda", default=OBJ_LOSS_LAMBDA, type=float,
                            help="The object loss lambda (weight).")

        return parser

    def __init__(self, model, num_classes, args):
        super().__init__()

        self.model = model
        self.num_classes = num_classes

        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.lr_decay_pow = args.lr_decay_pow
        self.focal_loss_scale = args.focal_loss_scale
        self.separation_loss = args.separation_loss
        self.separation_loss_lambda = args.separation_loss_lambda
        self.separation_loss_sky = args.separation_loss_sky
        self.pairwise_affinity_loss = args.pairwise_affinity_loss
        self.pairwise_affinity_loss_lambda = args.pairwise_affinity_loss_lambda
        self.pairwise_affinity_loss_tau = args.pairwise_affinity_loss_tau
        self.object_loss = args.object_loss
        self.object_loss_lambda = args.object_loss_lambda
        self.object_loss_pa = args.object_loss_pa
        self.object_loss_proj = args.object_loss_proj
        self.object_loss_aux = args.object_loss_aux

        self.val_accuracy = PixelAccuracy(num_classes)
        self.val_iou_0 = ClassIoU(0, num_classes)
        self.val_iou_1 = ClassIoU(1, num_classes)
        self.val_iou_2 = ClassIoU(2, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output['out']

    def training_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features)

        fl = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        separation_loss = torch.tensor(0.0)
        if self.separation_loss:
            separation_loss = water_obstacle_separation_loss(
                out['aux'], labels['segmentation'], include_sky=self.separation_loss_sky)

        pa_loss = torch.tensor(0.0)
        if self.pairwise_affinity_loss:
            ignore_mask = None
            if self.object_loss is not None:
                ignore_mask = labels['objects'].max(1).values
            pa_loss = pairwise_affinity_loss(out['out'], labels['segmentation'], labels['pa_similarity'],
                                             tau=self.pairwise_affinity_loss_tau, target_scale=self.focal_loss_scale, ignore_mask=ignore_mask)

        separation_loss = self.separation_loss_lambda * separation_loss
        pa_loss = self.pairwise_affinity_loss_lambda * pa_loss
        bg_loss = fl + separation_loss + pa_loss

        obj_loss_o = torch.tensor(0.0)
        obj_loss_pa = torch.tensor(0.0)
        obj_loss_aux = torch.tensor(0.0)
        obj_loss = torch.tensor(0.0)
        if self.object_loss:
            if self.object_loss_proj:
                obj_loss_o = object_projection_loss(out['out'], labels['objects'], labels['n_objects'], target_scale=self.focal_loss_scale,
                                                    normalize=True, reduce='sum')

            if self.object_loss_pa:
                obj_loss_pa = object_pairwise_affinity_loss(out['out'], labels['objects'], labels['n_objects'], labels['pa_similarity'],
                                                            tau=self.pairwise_affinity_loss_tau, normalize=True, reduce='sum')

            if self.object_loss_aux:
                obj_loss_aux = object_focal_loss(out['out'], labels['objects'], labels['n_objects'], labels['instance_seg'],
                                                normalize=True, reduce='sum')

            obj_loss = obj_loss_o + obj_loss_pa + obj_loss_aux

        loss = bg_loss + obj_loss

        # log losses
        self.log('train/loss', loss.item())
        self.log('train/bg/focal_loss', fl.item())
        self.log('train/bg/separation_loss', separation_loss.item())
        self.log('train/bg/pa_loss', pa_loss.item())
        self.log('train/bg_loss', bg_loss.item())
        self.log('train/obj/proj_loss', obj_loss_o.item())
        self.log('train/obj/pa_loss', obj_loss_pa.item())
        self.log('train/obj/aux_loss', obj_loss_aux.item())
        self.log('train/obj_loss', obj_loss.item())

        return loss

    def training_epoch_end(self, outputs):
        # Bugfix
        pass

    def validation_step(self, batch, batch_idx):
        features, labels = batch

        out = self.model(features)

        loss = focal_loss(out['out'], labels['segmentation'], target_scale=self.focal_loss_scale)

        # Log loss
        self.log('val/loss', loss.item())

        # Metrics
        labels_size = (labels['segmentation'].size(2), labels['segmentation'].size(3))
        logits = TF.resize(out['out'], labels_size, interpolation=Image.BILINEAR)
        preds = logits.argmax(1)

        # Create hard labels from soft
        labels_hard = labels['segmentation'].argmax(1)
        ignore_mask = labels['segmentation'].sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask

        self.val_accuracy(preds, labels_hard)
        self.val_iou_0(preds, labels_hard)
        self.val_iou_1(preds, labels_hard)
        self.val_iou_2(preds, labels_hard)

        self.log('val/accuracy', self.val_accuracy)
        self.log('val/iou/obstacle', self.val_iou_0)
        self.log('val/iou/water', self.val_iou_1)
        self.log('val/iou/sky', self.val_iou_2)

        return {'loss': loss, 'preds': preds}

    def configure_optimizers(self):
        # Separate parameters for different LRs
        encoder_parameters = []
        decoder_w_parameters = []
        decoder_b_parameters = []
        for name, parameter in self.model.named_parameters():
            if name.startswith('backbone'):
                encoder_parameters.append(parameter)
            elif 'weight' in name:
                decoder_w_parameters.append(parameter)
            else:
                decoder_b_parameters.append(parameter)

        optimizer = torch.optim.RMSprop([
            {'params': encoder_parameters, 'lr': self.learning_rate},
            {'params': decoder_w_parameters, 'lr': self.learning_rate * 10},
            {'params': decoder_b_parameters, 'lr': self.learning_rate * 20},
        ], momentum=self.momentum, alpha=0.9, weight_decay=self.weight_decay)

        # Decaying LR function
        lr_fn = lambda epoch: (1 - epoch/self.epochs) ** self.lr_decay_pow

        # Decaying learning rate (updated each epoch)
        scheduler = LambdaLR(optimizer, lr_fn)

        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # Export the model weights
        checkpoint['model'] = self.model.state_dict()
