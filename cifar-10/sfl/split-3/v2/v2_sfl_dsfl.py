import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torchvision import datasets, transforms
import os
import copy
import time
import itertools
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# --- PART 1: Model Architectures ---
# (Shared by both SFLV2 and DSFLV2)
# ==========================================


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(object):
    def __init__(self, block, depth, num_classes):
        super(CifarResNet, self).__init__()
        assert (depth - 2) % 6 == 0, "depth should be one of 20, 32, 44, 56, 110"
        layer_blocks = (depth - 2) // 6
        self.num_classes = num_classes
        self.inplanes = 16
        self.conv_1_3x3 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_1 = nn.BatchNorm2d(16)
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)

        self.layers = [self.conv_1_3x3, self.bn_1, nn.ReLU(True)]
        self.layers += list(self.stage_1.children())
        self.layers += list(self.stage_2.children())
        self.layers += list(self.stage_3.children())
        self.layers += [self.avgpool]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class CifarResNetDDG(nn.Module):
    def __init__(self, model, layers, split_id, num_splits, num_classes_aux):
        super(CifarResNetDDG, self).__init__()
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class auxillary_classifier2(nn.Module):
    def __init__(self, input_features, in_size, num_classes, n_lin=3, mlp_layers=3):
        super(auxillary_classifier2, self).__init__()
        self.avg_size = 2
        self.in_size = in_size
        self.feature_size = input_features
        self.n_lin = n_lin

        self.blocks = nn.ModuleList([])
        for n in range(n_lin):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_features,
                        input_features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(input_features),
                    nn.ReLU(True),
                )
            )

        if mlp_layers > 0:
            mlp_feat = input_features * 4
            layers = []
            for l in range(mlp_layers):
                if l == 0:
                    in_feat = input_features
                else:
                    in_feat = mlp_feat
                layers += [
                    nn.Linear(in_feat, mlp_feat),
                    nn.BatchNorm1d(mlp_feat),
                    nn.ReLU(True),
                ]
            layers += [nn.Linear(mlp_feat, num_classes)]
            self.classifier = nn.Sequential(*layers)
            self.mlp = True
        else:
            self.mlp = False
            self.classifier = nn.Linear(input_features, num_classes)

    def forward(self, x):
        out = x
        if out.shape[2] > 2:
            out = F.adaptive_avg_pool2d(out, (2, 2))
        out = out.view(out.size(0), -1)
        if self.classifier[0].in_features != out.shape[1]:
            self.classifier[0] = nn.Linear(
                out.shape[1], self.classifier[0].out_features
            ).to(x.device)
        out = self.classifier(out)
        return out


class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks

    def forward(self, x, n, upto=False):
        if upto:
            for i in range(n + 1):
                x = self.forward(x, i, upto=False)
            return x
        out = self.blocks[n](x)
        return out


class Net(nn.Module):
    def __init__(self, depth=110, num_classes=10, num_splits=3):
        super(Net, self).__init__()
        self.blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])

        model = CifarResNet(ResNetBasicblock, depth, num_classes)
        len_layers = len(model.layers)
        split_depth = math.ceil(len_layers / num_splits)

        for splits_id in range(num_splits):
            left_idx = splits_id * split_depth
            right_idx = (splits_id + 1) * split_depth
            if right_idx > len_layers:
                right_idx = len_layers

            net = CifarResNetDDG(
                model,
                model.layers[left_idx:right_idx],
                splits_id,
                num_splits,
                num_classes,
            )
            self.blocks.append(net)

            if splits_id < num_splits - 1:
                self.auxillary_nets.append(
                    auxillary_classifier2(
                        input_features=64,
                        in_size=8,
                        num_classes=num_classes,
                        n_lin=3,
                        mlp_layers=3,
                    )
                )

        self.auxillary_nets.append(model.classifier)
        self.main_cnn = rep(self.blocks)

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        if n == len(self.auxillary_nets) - 1:
            representation = representation.view(representation.size(0), -1)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation


# ==========================================
# --- PART 2: Utils and Metrics ---
# (Shared by both SFLV2 and DSFLV2)
# ==========================================


class MetricsTracker:
    def __init__(self, scenario_name, log_file):
        self.scenario = scenario_name
        self.log_file = log_file
        self.reset()

        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                header = (
                    "Scenario,Epoch,TrainAcc,TestAcc,Loss,CommFwdBytes,CommBwdBytes,"
                    "ClientTimeFwd,ClientTimeBwd,ServerTimeFwd,ServerTimeBwd,CommTimeWall,TotalTime,"
                    "PeakGPUMemMB\n"
                )
                f.write(header)

    def reset(self):
        self.comm_fwd_bytes = 0
        self.comm_bwd_bytes = 0
        self.client_time_fwd = 0.0
        self.client_time_bwd = 0.0
        self.server_time_fwd = 0.0
        self.server_time_bwd = 0.0
        self.training_time = 0
        self.peak_gpu_mem = 0

    def log_comm(self, tensor, direction="fwd"):
        if tensor is None:
            return
        size_bytes = tensor.numel() * 4
        if direction == "fwd":
            self.comm_fwd_bytes += size_bytes
        else:
            self.comm_bwd_bytes += size_bytes

    def update_gpu_stats(self):
        if torch.cuda.is_available():
            current_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            if current_peak > self.peak_gpu_mem:
                self.peak_gpu_mem = current_peak

    def save_epoch(self, epoch, train_acc, test_acc, loss):
        total_compute = (
            self.client_time_fwd
            + self.client_time_bwd
            + self.server_time_fwd
            + self.server_time_bwd
        )
        comm_wall_time = max(0, self.training_time - total_compute)
        self.update_gpu_stats()

        with open(self.log_file, "a") as f:
            f.write(
                f"{self.scenario},{epoch},{train_acc:.4f},{test_acc:.4f},{loss:.4f},"
                f"{self.comm_fwd_bytes},{self.comm_bwd_bytes},"
                f"{self.client_time_fwd:.4f},{self.client_time_bwd:.4f},"
                f"{self.server_time_fwd:.4f},{self.server_time_bwd:.4f},"
                f"{comm_wall_time:.4f},{self.training_time:.4f},{self.peak_gpu_mem:.2f}\n"
            )

        print(
            f"[{self.scenario}] Epoch {epoch} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | "
            f"Time: {self.training_time:.1f}s | GPU: {self.peak_gpu_mem:.0f}MB"
        )


def get_cifar10_loaders(num_clients, batch_size, alpha=None):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    targets = np.array(trainset.targets)
    class_indices = {}
    for idx in range(len(trainset)):
        label = int(targets[idx])
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    # ============================================================
    # INDEX ASSIGNMENT — IID path upgraded to stratified split.
    # ============================================================
    if alpha is None:
        # --- Stratified IID split (matches CIFAR standard) ---
        for label in sorted(class_indices.keys()):
            idxs = class_indices[label]
            np.random.shuffle(idxs)
            per_client = len(idxs) // num_clients
            for i in range(num_clients):
                client_indices[i].extend(idxs[i * per_client : (i + 1) * per_client])
    else:
        # --- NON-IID: Dirichlet(alpha) partition ---
        for label in sorted(class_indices.keys()):
            idxs = np.array(class_indices[label])
            np.random.shuffle(idxs)
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
            counts = (proportions * len(idxs)).astype(int)
            counts[-1] = len(idxs) - counts[:-1].sum()
            start = 0
            for i in range(num_clients):
                end = start + counts[i]
                client_indices[i].extend(idxs[start:end].tolist())
                start = end

    client_loaders = []
    for i in range(num_clients):
        subset = torch.utils.data.Subset(trainset, client_indices[i])
        loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        client_loaders.append(loader)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    print(f"Dataset: Cifar 10 | Images/Client: {len(client_indices[0])}")
    return client_loaders, test_loader


def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        return correct / target.size(0) * 100.0


def reset_env():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    import gc

    gc.collect()


def FedAvg(w_list):
    """Federated Averaging: uniform average of state_dicts."""
    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg.keys():
        for i in range(1, len(w_list)):
            w_avg[k] = w_avg[k] + w_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w_list))
    return w_avg


def evaluate(model, test_loader):
    """Evaluate using end-to-end forward pass through all blocks + final classifier."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            x = data
            for i in range(len(model.blocks)):
                x = model.blocks[i](x)
            x = x.view(x.size(0), -1)
            output = model.auxillary_nets[-1](x)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


# ==========================================
# --- PART 3A: SFLV2 Training Logic ---
# Server-side model is shared (single copy),
# updated in-place sequentially per client.
# No server-side FedAvg. Client-side FedAvg
# remains the same as SFLV1.
# ==========================================

ROUNDS = 100
LOCAL_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_sflv2(client_loaders, test_loader, num_splits, tracker):
    """
    SplitFed Learning V2 (SFLV2).
    Server-side model is a single shared instance updated sequentially
    by each client's smashed data (no server-side FedAvg).
    Client order is randomized each round.
    Client-side FedAvg is applied as in SFLV1.
    Communication: forward activations + backward gradients.
    """
    num_clients = len(client_loaders)
    global_model = Net(depth=110, num_classes=10, num_splits=num_splits).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        dummy = torch.randn(2, 3, 32, 32).to(DEVICE)
        x = dummy
        for i in range(num_splits):
            x = global_model.blocks[i](x)
        x = x.view(x.size(0), -1)
        global_model.auxillary_nets[-1](x)

    client_side_keys = set()
    server_side_keys = set()
    for k in global_model.state_dict().keys():
        if k.startswith("blocks.0.") or k.startswith("main_cnn.blocks.0."):
            client_side_keys.add(k)
        else:
            server_side_keys.add(k)

    print(f"\n--- Starting SFLV2 | Clients: {num_clients} | Splits: {num_splits} ---")
    print(
        f"    Client-side keys: {len(client_side_keys)} | Server-side keys: {len(server_side_keys)}"
    )
    print(f"    Server-side FedAvg: NONE (single model, sequential updates)")

    for round_num in range(1, ROUNDS + 1):
        tracker.reset()
        start_time_epoch = time.time()

        w_client_side_all = []

        total_loss = 0
        total_acc = 0
        total_samples = 0

        client_order = list(range(num_clients))
        random.shuffle(client_order)

        server_params = []
        for s in range(1, num_splits):
            server_params += list(global_model.blocks[s].parameters())
        server_params += list(global_model.auxillary_nets[-1].parameters())
        opt_server = optim.SGD(server_params, lr=0.001, momentum=0.9, weight_decay=1e-4)

        for client_id in client_order:
            loader = client_loaders[client_id]

            local_model = copy.deepcopy(global_model)
            local_model.train()
            global_model.train()

            client_params = list(local_model.blocks[0].parameters())
            opt_client = optim.SGD(
                client_params, lr=0.001, momentum=0.9, weight_decay=1e-4
            )

            for local_ep in range(LOCAL_EPOCHS):
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    opt_client.zero_grad()
                    opt_server.zero_grad()

                    # --- CLIENT FORWARD ---
                    t0 = time.time()
                    client_out = local_model.blocks[0](data)
                    client_out_var = client_out.detach().requires_grad_(True)
                    tracker.client_time_fwd += time.time() - t0
                    tracker.log_comm(client_out, "fwd")

                    # --- SERVER FORWARD (global_model, shared single instance) ---
                    t1 = time.time()
                    server_out = client_out_var
                    for s in range(1, num_splits):
                        server_out = global_model.blocks[s](server_out)
                    server_out_flat = server_out.view(server_out.size(0), -1)
                    final_pred = global_model.auxillary_nets[-1](server_out_flat)
                    loss = criterion(final_pred, target)
                    tracker.server_time_fwd += time.time() - t1

                    # --- SERVER BACKWARD (updates global_model in-place) ---
                    t2 = time.time()
                    loss.backward()
                    grad_at_cut = client_out_var.grad
                    opt_server.step()
                    tracker.server_time_bwd += time.time() - t2
                    tracker.log_comm(grad_at_cut, "bwd")

                    # --- CLIENT BACKWARD ---
                    t3 = time.time()
                    if grad_at_cut is not None:
                        client_out.backward(grad_at_cut)
                    opt_client.step()
                    tracker.client_time_bwd += time.time() - t3

                    total_loss += loss.item() * data.size(0)
                    total_acc += accuracy(final_pred, target) * data.size(0)
                    total_samples += data.size(0)

            local_sd = local_model.state_dict()
            w_client_side_all.append({k: local_sd[k].clone() for k in client_side_keys})

        w_avg_client = FedAvg(w_client_side_all)

        global_sd = global_model.state_dict()
        global_sd.update(w_avg_client)
        global_model.load_state_dict(global_sd)

        tracker.training_time = time.time() - start_time_epoch
        test_acc = evaluate(global_model, test_loader)
        train_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        tracker.save_epoch(round_num, train_acc, test_acc, avg_loss)


# ==========================================
# --- PART 3B: DSFLV2 Training Logic ---
# Decoupled SplitFed V2: client trains block 0
# with local auxiliary head (greedy, BP-free).
# Server uses single shared model, updated
# sequentially. No backward communication.
# No server-side FedAvg.
# ==========================================


def train_dsflv2(client_loaders, test_loader, num_splits, tracker):
    """
    Decoupled SplitFed Learning V2 (DSFLV2).
    Client trains block 0 + aux[0] with greedy local loss (DGL-style).
    Server trains blocks 1..N-1 + final classifier on detached activations,
    using a single shared model updated sequentially (no server-side FedAvg).
    Communication: forward activations ONLY.
    """
    num_clients = len(client_loaders)
    global_model = Net(depth=110, num_classes=10, num_splits=num_splits).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        dummy = torch.randn(2, 3, 32, 32).to(DEVICE)
        x = dummy
        for i in range(num_splits):
            x = global_model.blocks[i](x)
        x = x.view(x.size(0), -1)
        global_model.auxillary_nets[-1](x)

    with torch.no_grad():
        dummy = torch.randn(2, 3, 32, 32).to(DEVICE)
        feat = global_model.blocks[0](dummy)
        global_model.auxillary_nets[0](feat)

    client_side_keys = set()
    server_side_keys = set()
    for k in global_model.state_dict().keys():
        if (
            k.startswith("blocks.0.")
            or k.startswith("main_cnn.blocks.0.")
            or k.startswith("auxillary_nets.0.")
        ):
            client_side_keys.add(k)
        else:
            server_side_keys.add(k)

    print(f"\n--- Starting DSFLV2 | Clients: {num_clients} | Splits: {num_splits} ---")
    print(
        f"    Client-side keys: {len(client_side_keys)} | Server-side keys: {len(server_side_keys)}"
    )
    print(f"    Backward communication: NONE (fully decoupled)")
    print(f"    Server-side FedAvg: NONE (single model, sequential updates)")

    for round_num in range(1, ROUNDS + 1):
        tracker.reset()
        start_time_epoch = time.time()

        w_client_side_all = []

        total_loss = 0
        total_acc = 0
        total_samples = 0

        client_order = list(range(num_clients))
        random.shuffle(client_order)

        server_params = []
        for s in range(1, num_splits):
            server_params += list(global_model.blocks[s].parameters())
        server_params += list(global_model.auxillary_nets[-1].parameters())
        opt_server = optim.SGD(server_params, lr=0.001, momentum=0.9, weight_decay=1e-4)

        for client_id in client_order:
            loader = client_loaders[client_id]

            local_model = copy.deepcopy(global_model)
            local_model.train()
            global_model.train()

            client_params = list(
                itertools.chain(
                    local_model.blocks[0].parameters(),
                    local_model.auxillary_nets[0].parameters(),
                )
            )
            opt_client = optim.SGD(
                client_params, lr=0.001, momentum=0.9, weight_decay=1e-4
            )

            for local_ep in range(LOCAL_EPOCHS):
                for batch_idx, (data, target) in enumerate(loader):
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    # --- CLIENT SIDE: Greedy local training (DGL-style) ---
                    opt_client.zero_grad()

                    t0 = time.time()
                    client_out = local_model.blocks[0](data)
                    aux_pred = local_model.auxillary_nets[0](client_out)
                    tracker.client_time_fwd += time.time() - t0

                    t1 = time.time()
                    aux_loss = criterion(aux_pred, target)
                    aux_loss.backward()
                    opt_client.step()
                    tracker.client_time_bwd += time.time() - t1

                    tracker.log_comm(client_out, "fwd")

                    # --- SERVER SIDE: Training on detached input (global_model, shared) ---
                    opt_server.zero_grad()

                    t2 = time.time()
                    server_input = client_out.detach()
                    server_out = server_input
                    for s in range(1, num_splits):
                        server_out = global_model.blocks[s](server_out)
                    server_out_flat = server_out.view(server_out.size(0), -1)
                    final_pred = global_model.auxillary_nets[-1](server_out_flat)
                    server_loss = criterion(final_pred, target)
                    tracker.server_time_fwd += time.time() - t2

                    t3 = time.time()
                    server_loss.backward()
                    opt_server.step()
                    tracker.server_time_bwd += time.time() - t3

                    total_loss += server_loss.item() * data.size(0)
                    total_acc += accuracy(final_pred, target) * data.size(0)
                    total_samples += data.size(0)

            local_sd = local_model.state_dict()
            w_client_side_all.append({k: local_sd[k].clone() for k in client_side_keys})

        w_avg_client = FedAvg(w_client_side_all)

        global_sd = global_model.state_dict()
        global_sd.update(w_avg_client)
        global_model.load_state_dict(global_sd)

        tracker.training_time = time.time() - start_time_epoch
        test_acc = evaluate(global_model, test_loader)
        train_acc = total_acc / total_samples
        avg_loss = total_loss / total_samples
        tracker.save_epoch(round_num, train_acc, test_acc, avg_loss)


# ==========================================
# --- PART 4: Comparative Visualization ---
# Generates side-by-side and overlay plots
# comparing SFLV2 vs DSFLV2 across all metrics.
# ==========================================


def generate_comparison_visualizations(log_file):
    df = pd.read_csv(log_file)
    if not os.path.exists("logs/plots"):
        os.makedirs("logs/plots")

    df["Method"] = df["Scenario"].apply(
        lambda s: "SFLV2" if s.startswith("SFLV2") else "DSFLV2"
    )
    df["Clients"] = df["Scenario"].apply(lambda s: s.split("_")[-1])

    # --- 1. Test Accuracy Convergence (overlay) ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, x="Epoch", y="TestAcc", hue="Scenario", style="Method", markers=True
    )
    plt.title("SFLV2 vs DSFLV2 - Test Accuracy Convergence")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("logs/plots/compare_test_accuracy.png")
    plt.close()

    # --- 2. Training Accuracy Convergence ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, x="Epoch", y="TrainAcc", hue="Scenario", style="Method", markers=True
    )
    plt.title("SFLV2 vs DSFLV2 - Training Accuracy Convergence")
    plt.ylabel("Training Accuracy (%)")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("logs/plots/compare_train_accuracy.png")
    plt.close()

    # --- 3. Loss Convergence ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, x="Epoch", y="Loss", hue="Scenario", style="Method", markers=True
    )
    plt.title("SFLV2 vs DSFLV2 - Loss Convergence")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("logs/plots/compare_loss.png")
    plt.close()

    # --- 4. Per-client-count comparison (grouped bar: peak test accuracy) ---
    peak_acc = df.groupby(["Method", "Clients"])["TestAcc"].max().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=peak_acc, x="Clients", y="TestAcc", hue="Method")
    plt.title("SFLV2 vs DSFLV2 - Peak Test Accuracy by Client Count")
    plt.ylabel("Peak Test Accuracy (%)")
    plt.xlabel("Number of Clients")
    plt.tight_layout()
    plt.savefig("logs/plots/compare_peak_accuracy_bar.png")
    plt.close()

    # --- 5. Communication Volume Comparison (grouped bar) ---
    df["TotalCommMB"] = (df["CommFwdBytes"] + df["CommBwdBytes"]) / 1e6
    comm_per_epoch = (
        df.groupby(["Method", "Clients"])["TotalCommMB"].mean().reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comm_per_epoch, x="Clients", y="TotalCommMB", hue="Method")
    plt.title("SFLV2 vs DSFLV2 - Avg Communication Volume per Epoch (MB)")
    plt.ylabel("Communication (MB)")
    plt.xlabel("Number of Clients")
    plt.tight_layout()
    plt.savefig("logs/plots/compare_comm_volume.png")
    plt.close()

    # --- 6. Fwd vs Bwd Communication Breakdown ---
    df["CommFwdMB"] = df["CommFwdBytes"] / 1e6
    df["CommBwdMB"] = df["CommBwdBytes"] / 1e6
    comm_breakdown = (
        df.groupby(["Method", "Clients"])[["CommFwdMB", "CommBwdMB"]]
        .mean()
        .reset_index()
    )
    comm_melted = comm_breakdown.melt(
        id_vars=["Method", "Clients"],
        value_vars=["CommFwdMB", "CommBwdMB"],
        var_name="Direction",
        value_name="MB",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=comm_melted,
        x="Clients",
        y="MB",
        hue="Method",
        style="Direction" if False else None,
    )
    g = sns.catplot(
        data=comm_melted,
        x="Clients",
        y="MB",
        hue="Method",
        col="Direction",
        kind="bar",
        height=5,
        aspect=1.2,
    )
    g.fig.suptitle("SFLV2 vs DSFLV2 - Fwd vs Bwd Communication", y=1.02)
    g.savefig("logs/plots/compare_comm_fwd_bwd.png")
    plt.close("all")

    # --- 7. Total Time per Epoch (boxplot) ---
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Clients", y="TotalTime", hue="Method")
    plt.title("SFLV2 vs DSFLV2 - Time per Epoch Distribution")
    plt.ylabel("Time (s)")
    plt.xlabel("Number of Clients")
    plt.tight_layout()
    plt.savefig("logs/plots/compare_time_boxplot.png")
    plt.close()

    # --- 8. Compute Time Breakdown (stacked bar, side by side) ---
    time_cols = [
        "ClientTimeFwd",
        "ClientTimeBwd",
        "ServerTimeFwd",
        "ServerTimeBwd",
        "CommTimeWall",
    ]
    time_summary = df.groupby("Scenario")[time_cols].mean()
    time_summary.plot(
        kind="bar",
        stacked=True,
        figsize=(14, 6),
        color=["#2196F3", "#64B5F6", "#FF5722", "#FF8A65", "#9E9E9E"],
    )
    plt.title("SFLV2 vs DSFLV2 - Compute Time Breakdown per Epoch")
    plt.ylabel("Time (s)")
    plt.xlabel("Scenario")
    plt.legend(
        ["Client Fwd", "Client Bwd", "Server Fwd", "Server Bwd", "Comm/Overhead"],
        loc="upper left",
        fontsize=8,
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("logs/plots/compare_time_breakdown.png")
    plt.close()

    # --- 9. Peak GPU Memory Comparison ---
    mem_summary = df.groupby(["Method", "Clients"])["PeakGPUMemMB"].max().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=mem_summary, x="Clients", y="PeakGPUMemMB", hue="Method", palette="viridis"
    )
    plt.title("SFLV2 vs DSFLV2 - Peak GPU Memory")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.xlabel("Number of Clients")
    plt.tight_layout()
    plt.savefig("logs/plots/compare_gpu_memory.png")
    plt.close()

    # --- 10. Final Summary Table ---
    print("\n" + "=" * 80)
    print("  SFLV2 vs DSFLV2 -- COMPARATIVE SUMMARY")
    print("=" * 80)
    table_stats = (
        df.groupby("Scenario")
        .agg(
            {
                "TrainAcc": "max",
                "TestAcc": "max",
                "Loss": "min",
                "PeakGPUMemMB": "max",
                "TotalTime": "sum",
                "CommFwdBytes": "mean",
                "CommBwdBytes": "mean",
                "ClientTimeFwd": "mean",
                "ClientTimeBwd": "mean",
                "ServerTimeFwd": "mean",
                "ServerTimeBwd": "mean",
                "CommTimeWall": "mean",
            }
        )
        .rename(
            columns={
                "TrainAcc": "Peak Train %",
                "TestAcc": "Peak Test %",
                "Loss": "Min Loss",
                "PeakGPUMemMB": "Peak Mem MB",
                "TotalTime": "Total Time s",
                "CommFwdBytes": "Avg Fwd Bytes/Ep",
                "CommBwdBytes": "Avg Bwd Bytes/Ep",
                "ClientTimeFwd": "Avg C-Fwd s",
                "ClientTimeBwd": "Avg C-Bwd s",
                "ServerTimeFwd": "Avg S-Fwd s",
                "ServerTimeBwd": "Avg S-Bwd s",
                "CommTimeWall": "Avg Overhead s",
            }
        )
    )
    print(table_stats.to_string())
    table_stats.to_csv("logs/sflv2_vs_dsflv2_summary.csv")

    comm_stats = df.groupby("Scenario").agg(
        {
            "CommFwdBytes": ["mean", "sum"],
            "CommBwdBytes": ["mean", "sum"],
        }
    )
    comm_stats.columns = [
        "Avg Fwd Bytes/Epoch",
        "Total Fwd Bytes",
        "Avg Bwd Bytes/Epoch",
        "Total Bwd Bytes",
    ]
    comm_stats["Avg Total MB/Epoch"] = (
        comm_stats["Avg Fwd Bytes/Epoch"] + comm_stats["Avg Bwd Bytes/Epoch"]
    ) / 1e6
    comm_stats["Grand Total MB"] = (
        comm_stats["Total Fwd Bytes"] + comm_stats["Total Bwd Bytes"]
    ) / 1e6
    print("\n--- Communication Comparison ---")
    print(comm_stats.to_string())
    comm_stats.to_csv("logs/sflv2_vs_dsflv2_comm.csv")

    print(f"\nAll comparison plots saved to: logs/plots/")
    print(f"Summary CSV: logs/sflv2_vs_dsflv2_summary.csv")
    print(f"Comm CSV:    logs/sflv2_vs_dsflv2_comm.csv")


# ==========================================
# --- PART 5: Main Execution ---
# Runs SFLV2 and DSFLV2 for each client count,
# writing to a single shared CSV, then
# generates comparative visualizations.
# ==========================================

if __name__ == "__main__":
    if not os.path.exists("logs"):
        os.makedirs("logs")
    log_path = "logs/sflv2_vs_dsflv2_results.csv"

    if os.path.exists(log_path):
        os.remove(log_path)

    num_splits = 3

    for num_clients in [10, 5, 1]:
        # --- SFLV2 ---
        reset_env()
        clients, test = get_cifar10_loaders(
            num_clients=num_clients, batch_size=128, alpha=0.5
        )
        tracker = MetricsTracker(f"SFLV2_1_{num_clients}", log_path)
        train_sflv2(clients, test, num_splits, tracker=tracker)

        # --- DSFLV2 ---
        reset_env()
        clients, test = get_cifar10_loaders(
            num_clients=num_clients, batch_size=128, alpha=0.5
        )
        tracker = MetricsTracker(f"DSFLV2_1_{num_clients}", log_path)
        train_dsflv2(clients, test, num_splits, tracker=tracker)

    print("\n" + "=" * 60)
    print("  All training completed. Generating comparison plots...")
    print("=" * 60)
    generate_comparison_visualizations(log_path)
    print("Done. Check 'logs/' folder.")
