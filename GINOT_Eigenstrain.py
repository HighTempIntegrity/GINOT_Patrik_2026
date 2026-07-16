# %%
import os
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
if __package__:
    from . import configs
    from .modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from .modules.UNets import UNet
    from .trainer import torch_trainer
    from .modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from .modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels
else:
    import configs
    from modules.point_encoding import PointCloudPerceiverChannelsEncoder
    from modules.UNets import UNet
    from trainer import torch_trainer
    from modules.transformer import SelfAttentionBlocks, MLP, ResidualCrossAttentionBlock
    from modules.point_position_embedding import PosEmbLinear, encode_position, position_encoding_channels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")


# %%

class Trunk(nn.Module):
    def __init__(self, branch, embed_dim=64, cross_attn_layers=4, num_heads=8,
                 in_channels=18, out_channels=6,
                 dropout=0.0, emd_version="nerf", padding_value=-10):
        super().__init__()
        self.padding_value = padding_value
        d = position_encoding_channels(emd_version)
        # self.Q_encoder = MLP(embed_dim, in_channels)
        self.Q_encoder = nn.Sequential(nn.Linear(d*in_channels, 2*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(2*embed_dim, 3*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(3*embed_dim, 2*embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(2*embed_dim, embed_dim)
                                       )
        self.branch = branch
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    width=embed_dim,
                    heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(cross_attn_layers)
            ]
        )
        self.output_proj = nn.Sequential(nn.Linear(embed_dim, 2*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(2*embed_dim, 3*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(3*embed_dim, 3*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(3*embed_dim, 2*embed_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(2*embed_dim, out_channels)
                                         )

    def forward(self, xyt, pc, sample_ids=None):
        # (B, latenc, embed_dim)
        latent = self.branch(pc, sample_ids=sample_ids)
        # (B,N,ndim)->(B,N,embed_dim)
        xyt = encode_position('nerf', position=xyt)
        x = self.Q_encoder(xyt)
        for block in self.resblocks:
            x = block(x, latent)  # (B, N, embed_dim)
        # (B, N, embed_dim)->(B, N, 1)
        x = self.output_proj(x)
        return x.squeeze(-1)


# %%

def NOTModelDefinition(branch_args, trunc_args):
    branch = PointCloudPerceiverChannelsEncoder(**branch_args)
    tot_num_params = sum(p.numel() for p in branch.parameters())
    trainable_params = sum(p.numel()
                           for p in branch.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of Geo encoder: {tot_num_params}, {trainable_params} of which are trainable")
    trunk = Trunk(branch, **trunc_args)
    tot_num_params = sum(p.numel() for p in trunk.parameters())
    trainable_params = sum(p.numel()
                           for p in trunk.parameters() if p.requires_grad)
    print(
        f"Total number of parameters of NOT model: {tot_num_params}, {trainable_params} of which are trainable")

    return trunk

# %%
def EvaluateForwardModel_test(trainer, test_loader, val_loader, train_loader):
    trainer.load_weights(device=device_cpu)


    def cal_l2_error(val_loader):
        y_pred, y_true = trainer.predict_forward(val_loader)
        error_s = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = np.linalg.norm(s_p-s_t)/np.linalg.norm(s_t)
            error_s.append(e_s)
        error_s = np.array(error_s)
        return error_s

    
    def cal_wmape_error(val_loader):
        y_pred, y_true = trainer.predict_forward(val_loader)
        error_wmape = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = (np.abs(s_p - s_t)).sum() / (np.abs(s_t)).sum()
            error_wmape.append(e_s)
        error_wmape = np.array(error_wmape)
        return error_wmape
    
 

    error_l2 = cal_l2_error(test_loader)
    error_wmape = cal_wmape_error(test_loader)
    print(
        f"\nMean L2 error for test data: {np.mean(error_l2)}, std: {np.std(error_l2)}")
    print(
        f"Mean wmape error for test data: {np.mean(error_wmape)}, std: {np.std(error_wmape)}")

    
    def predict_and_save_eigenstrains(test_loader):
        """
        Predicts strain tensor components and saves each component to a separate .txt file
        for each batch. Also saves the absolute error between predicted and true values
        component-wise, and prints summary error statistics.
    
        Output: 
        - Eigenstrains_Batch0_11.txt, Eigenstrains_Batch0_abs_error_11.txt, etc.
        """
    
        y_pred, y_true = trainer.predict_forward(test_loader)  # y_pred and y_true are lists of arrays, one per batch
    
        component_names = ["11", "22", "33", "12", "13", "23"]  # adjust based on model output
    
        for batch_idx, (batch_pred, batch_true) in enumerate(zip(y_pred, y_true)):
            batch_pred = np.array(batch_pred)  # shape: (N, num_components)
            batch_true = np.array(batch_true)  # same shape
    
            print(f"\nBatch {batch_idx} prediction shape: {batch_pred.shape}")
    
            # Transpose to (num_components, N)
            batch_pred_T = batch_pred.T
            batch_true_T = batch_true.T
            
            numerator = np.linalg.norm(batch_pred - batch_true)
            denominator = np.linalg.norm(batch_true)
            batch_l2 = numerator / denominator
            
            batch_wmape = np.abs(batch_pred - batch_true).sum() / np.abs(batch_true).sum()

            print(f"Batch {batch_idx} - Batch-level Errors:")
            print(f"  L2 Error   : {batch_l2:.4e}")
            print(f"  WMAPE      : {batch_wmape:.4e}")
    
            for i, comp_name in enumerate(component_names):
                # Save predicted values
                pred_filename = f"Eigenstrains_Batch{batch_idx}_{comp_name}.txt"
                np.savetxt(pred_filename, batch_pred_T[i], fmt="%.6e")
                print(f"\nSaved {pred_filename}")
    
                # Compute and save absolute error
                abs_error = np.abs(batch_pred_T[i] - batch_true_T[i])
                error_filename = f"Abs_error_Batch{batch_idx}_{comp_name}.txt"
                np.savetxt(error_filename, abs_error, fmt="%.6e")
                print(f"Saved {error_filename}")
    
                # Compute and print error statistics
                mean_err = np.mean(abs_error)
                std_err = np.std(abs_error)
                max_err = np.max(abs_error)
    
                print(f"Batch {batch_idx}, Component {comp_name} - Mean Abs Error: {mean_err:.3e}, Std: {std_err:.3e}, Max: {max_err:.3e}")
    
    predict_and_save_eigenstrains(test_loader)


# %%
def EvaluateForwardModel(trainer, test_loader, val_loader, train_loader):
    trainer.load_weights(device=device)


    def cal_l2_error(val_loader):
        y_pred, y_true = trainer.predict(val_loader)
        error_s = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = np.linalg.norm(s_p-s_t)/np.linalg.norm(s_t)
            error_s.append(e_s)
        error_s = np.array(error_s)
        return error_s
    
    def cal_l1_error(val_loader):
        y_pred, y_true = trainer.predict(val_loader)
        error_l1 = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_l1 = (np.abs(s_p - s_t) / np.abs(s_t)).mean()
            # print(e_l1)
            error_l1.append(e_l1)
        error_l1 = np.array(error_l1)
        return error_l1
    
    def cal_ae_error(val_loader):
        y_pred, y_true = trainer.predict(val_loader)
        error_ae = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = (np.abs(s_p - s_t)).mean()
            error_ae.append(e_s)
        error_ae = np.array(error_ae)
        return error_ae
    
    def cal_wmape_error(val_loader):
        y_pred, y_true = trainer.predict(val_loader)
        error_wmape = []
        for y_p, y_t in zip(y_pred, y_true):
            s_p, s_t = y_p[:], y_t[:]
            e_s = (np.abs(s_p - s_t)).sum() / (np.abs(s_t)).sum()
            error_wmape.append(e_s)
        error_wmape = np.array(error_wmape)
        return error_wmape
    
    def cal_r_correlation(val_loader,i):
        y_pred, y_true = trainer.predict(val_loader)

        # Flatten
        y_pred_flat = np.concatenate([y_p.flatten() for y_p in y_pred])
        y_true_flat = np.concatenate([y_t.flatten() for y_t in y_true])

        r = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
        
        mode = "Training" if i == 0 else "Testing"
        
        # Create scatter plot
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, edgecolor='k')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"R Correlation Plot - {mode}")

        # Plot identity line (perfect prediction)
        lims = [min(y_true_flat.min(), y_pred_flat.min()), max(y_true_flat.max(), y_pred_flat.max())]
        plt.plot(lims, lims, 'r--', label='Ideal fit (y = x)')
    
        # Annotate R
        plt.text(0.05, 0.95, f"Pearson R = {r:.3f}", transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # Save figure
        filename = f"r_correlation_plot_{mode.lower()}.png"
        plt.savefig(filename, dpi=300)

        plt.show()
        return r
    
    

    error_s = cal_l2_error(val_loader)
    sort_idx = np.argsort(error_s)
    idx_best = sort_idx[0]
    idx_32perc = sort_idx[int(len(sort_idx)*0.32)]
    idx_63perc = sort_idx[int(len(sort_idx)*0.63)]
    idx_99perc = sort_idx[int(len(sort_idx)*0.99)]
    index_list = [idx_best, idx_32perc, idx_63perc, idx_99perc]
    labels = ["\nBest validation", "32th percentile", "63th percentile", "99th percentile"]
    for label, idx in zip(labels, index_list):
        print(f"{label} L2 error: {error_s[idx]}")

    error_l2 = cal_l2_error(train_loader)
    error_wmape = cal_wmape_error(train_loader)
    print(
        f"\nMean L2 error for training data: {np.mean(error_l2)}, std: {np.std(error_l2)}")
    print(
        f"Mean wMAPE error for training data: {np.mean(error_wmape)}, std: {np.std(error_wmape)}")


    error_l2 = cal_l2_error(val_loader)
    error_wmape = cal_wmape_error(val_loader)
    print(
        f"\nMean L2 error for validation data: {np.mean(error_l2)}, std: {np.std(error_l2)}")
    print(
        f"Mean wmape error for validation data: {np.mean(error_wmape)}, std: {np.std(error_wmape)}")


   

def TrainNOTModel(NTO_model, filebase, train_flag, epochs, lr, bs_train, bs_val, bs_test, window_size=None):

    train_loader, val_loader, test_loader, s_inverse, pc_inverse, vert_inverse = configs.LoadDataEigenstrain(
        bs_train=bs_train, bs_val=bs_val, bs_test=bs_test)
    

    class TRAINER(torch_trainer.TorchTrainer):
        def __init__(self, models, device, filebase):
            super().__init__(models, device, filebase)

        def evaluate_losses(self, data):
            pc = data[0].to(self.device)
            xyt = data[1].to(self.device)
            y_true = data[2].to(self.device)
            sample_ids = data[3].to(self.device)
            mask = (y_true != self.models[0].padding_value).float()
            y_pred = self.models[0](xyt, pc, sample_ids)
            loss = nn.MSELoss(reduction='none')(y_true, y_pred)
            loss = (loss*mask).sum()/(mask.sum()+1)
            loss_dic = {"loss": loss.item()}

            # Append current loss to file
            with open("training_loss_log.txt", "a") as f:
                 f.write(f"Loss: {loss.item():.6f}\n")

            return loss, loss_dic

        def predict(self, data_loader):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    pc = data[0].to(self.device)
                    xyt = data[1].to(self.device)
                    y_true_batch = data[2].to(self.device)
                    mask = (y_true_batch != self.models[0].padding_value)
                    pred = self.models[0](xyt, pc)
                    pred = s_inverse(pred)
                    y_true_batch = s_inverse(y_true_batch)
                    pred = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                            for x, i in zip(pred, mask)]
                    y_true_batch = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                                    for x, i in zip(y_true_batch, mask)]

                    y_pred = y_pred+pred
                    y_true = y_true+y_true_batch

            return y_pred, y_true

        def predict_forward(self, data_loader):
            y_pred = []
            y_true = []
            self.models[0].eval()
            with torch.no_grad():
                for data in data_loader:
                    pc = data[0].to(self.device)
                    xyt = data[1].to(self.device)
                    y_true_batch = data[2].to(self.device)
                    mask = (y_true_batch != self.models[0].padding_value)
                    pred = self.models[0](xyt, pc)
                    pred = s_inverse(pred)
                    y_true_batch = s_inverse(y_true_batch)
                    pred = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                            for x, i in zip(pred, mask)]
                    y_true_batch = [x[i].view(-1, *i.shape[1:]).cpu().detach().numpy()
                                    for x, i in zip(y_true_batch, mask)]

                    y_pred = y_pred+pred
                    y_true = y_true+y_true_batch

            return y_pred, y_true


    trainer = TRAINER(
        NTO_model, device, filebase)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="loss", save_best_only=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=20)
    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="val_loss",
        window_size=window_size,
        sequence_idx=[1, 2],
    )
    if train_flag == "continue":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    h = trainer.fit(train_loader, val_loader=val_loader,
                    epochs=epochs, print_freq=1)
    trainer.save_logs()
    
    EvaluateForwardModel(trainer, test_loader, val_loader, train_loader)
    
    
    for m in trainer.models:
        m.to(device_cpu)
    
    trainer.device = device_cpu
    
    EvaluateForwardModel_test(trainer, test_loader, val_loader, train_loader)
    return trainer


def LoadModel(filebase, branch_args, trunk_args):
    NTO_model = NOTModelDefinition(branch_args, trunk_args)
    model_path = os.path.join(filebase, "model.ckpt")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    NTO_model.load_state_dict(state_dict)
    NTO_model.to(device)
    NTO_model.eval()
    return NTO_model


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs_train", type=int, default=4)
    parser.add_argument("--bs_val", type=int, default=4)
    parser.add_argument("--bs_test", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=None)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs_ginot = configs.Eigenstrain_GINOT_configs()

    filebase = configs_ginot["filebase"]
    trunk_args = configs_ginot["trunk_args"]
    branch_args = configs_ginot["branch_args"]
    print(configs_ginot)


    NTO_model = NOTModelDefinition(branch_args, trunk_args)

    trainer = TrainNOTModel(NTO_model, filebase, args.train_flag,
                            epochs=args.epochs, lr=args.learning_rate, 
                            bs_train=args.bs_train,bs_val=args.bs_val,
                            bs_test=args.bs_test, window_size=args.window_size)
    print(filebase, " training finished")

# %%