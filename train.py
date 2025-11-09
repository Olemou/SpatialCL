from utils import *
from ddp.dist_util import is_dist, get_world_size,get_rank, wrap_model, get_model_device
from torch.utils.data import DataLoader, DistributedSampler
from  ddp.env import setup_env, seed_everything
from ddp.parse import parse_ddp_args,init_distributed_mode
from utils.dataloader import get_datasets_and_loaders, ThermalAugmentation, RgbAugmentation
from utils.config import ThermalAugConfig, RgbAugConfig
from model.vit import  VisionTransformer
from model.load_weight import load_pretrained_vit_weights
from utils.scheduler import get_optimizer, cosine_schedule
from spatialcl.uwcl import build_uwcl




def param_dataloader_init(
    root: str = None,
    dataset_class=None,
    modality: dict = {"rgb": True, "thermal": False},
):
    """Main training setup for distributed or single-node training."""
    # --- Environment setup ---
    setup_env()
    seed_everything()

    # --- Distributed setup ---
    args = parse_ddp_args()
    init_distributed_mode(args)

    # --- Select transformation based on modality ---
    if modality.get("rgb", False):
        transform = RgbAugmentation(RgbAugConfig()).transform
    elif modality.get("thermal", False):
        transform = ThermalAugmentation(ThermalAugConfig()).transform
    else:
        raise ValueError("Please specify at least one valid modality: 'rgb' or 'thermal'.")

    # --- Get Datasets ---
    train_dataset, val_dataset, test_dataset = get_datasets_and_loaders(
        root=root,
        dataset_class=dataset_class,
        transform=transform,
    )

    # --- Samplers (DDP-aware) ---
    def build_sampler(dataset):
        if is_dist():
            return DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True,
                drop_last=True
            )
        return None

    sampler_train = build_sampler(train_dataset)
    sampler_val = build_sampler(val_dataset)
    sampler_test = build_sampler(test_dataset)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler_test,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, args

def one_epoch_train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,   
):
    """One epoch training loop."""
    model.train()
    total_loss = 0.0
    for (x1, x2), labels, _ in train_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        optimizer.zero_grad()
        features1 = model(x1)
        features2 = model(x2)
        loss = criterion(features1, features2, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x1.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss  
# =====================
def one_eval_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
):
    """One epoch evaluation loop."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (x1, x2), labels, _ in val_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            features1 = model(x1)
            features2 = model(x2)
            loss = criterion(features1, features2, labels)

            total_loss += loss.item() * x1.size(0)

    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss 
# =====================

def main(
    root: str = None,
    dataset_class=None,
    vit_varaint: str = "base",
    modality: dict = {"rgb": True, "thermal": False},
):
   
    """Main training setup for distributed or single-node training."""
    # --- DataLoaders & DDP setup ---
    train_loader, val_loader, test_loader, args = param_dataloader_init(
        root=root,
        dataset_class=dataset_class,
        modality=modality,
    )

    # --- Model, criterion, optimizer ---
    model = wrap_model(VisionTransformer(variant=vit_varaint))
    device = get_model_device(model)
    weighted_model = load_pretrained_vit_weights(
        custom_model=model,
        model_size=vit_varaint,
        device=device
    )
    optimizer = get_optimizer(model=weighted_model)
    criterion = build_uwcl

          
   

    # --- Training loop ---
    for epoch in range(args.epochs):
        cosine_schedule(epoch = epoch, max_epochs=  args.epochs, warmup_epochs= args.warmup_epochs)
        
        train_loss = one_epoch_train(model, train_loader, criterion, optimizer, device)
        val_loss = one_eval_epoch(model, val_loader, criterion, device)

       