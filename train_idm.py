import os
import random
import numpy as np
import torch
import wandb
import argparse
from absl import app
from tqdm import tqdm
from datetime import datetime

from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from idm.dataset import DataSet
from idm.idm import *
from idm.loss import AdaptiveLoss, WeightedSmoothL1Loss
from idm.preprocessor import DinoPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Train IDM")
    parser.add_argument("--load_from", type=str, default=None, help="Load from path")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Wandb mode")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_transform", action="store_true", default=False, help="Use transform")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches to prefetch")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path of the dataset")
    parser.add_argument("--num_iterations", type=int, default=150000, help="Number of iterations")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Intervals of evaluation. ")
    parser.add_argument("--run_name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), help="Run name")
    parser.add_argument("--save_dir", type=str, default="output", help="Save dir")
    parser.add_argument("--ratio_eval", type=float, default=0.05, help="Ratio of data for validation, but eval_dataset_size is at most 10000")
    parser.add_argument("--model_name", type=str, default="direction_aware", help="Choose a suitable model.")
    parser.add_argument("--dinov2_name", type=str, default="facebook/dinov2-base", help="DINOv2 name.")
    parser.add_argument("--freeze_dinov2", action="store_true", default=False, help="Freeze DINOv2")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["constant", "cosine"], help="Learning rate scheduler type")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--test_dataset_path", type=str, default=None, help="Path of the test dataset")
    parser.add_argument("--eval_only", action="store_true", default=False, help="Only run evaluation on val and test sets")
    parser.add_argument("--use_normalization", action="store_true", default=False, help="Use mean/std normalization")
    parser.add_argument("--load_mp4", action="store_true", default=True, help="load the data in mp4 format to save memory")
    parser.add_argument("--learning_dim", type=str, default=None, help="Comma-separated list of dimensions to learn (e.g. '0,1,2,3'). If None, learn all dimensions.")
    parser.add_argument("--seed",type=int, default=0, help="seed")
    args = parser.parse_args()
    return args

def collate_fn(batch, preprocessor: DinoPreprocessor):
    # batch is a list of tuples (pos, image), pos is [B, 14], \del image is [B, 3, 518, 518] tensor \del
    # image is a pil image: [H, W, C]
    # concatenate all pos and images from list to tensor
    pos, images = zip(*batch)
    # preprocess images
    images = preprocessor.process_batch(images)  # [B, 3, 518, 518]
    pos = torch.stack(pos)  # [B, 14]
    # flip images and pos
    images, pos = preprocessor.flip_images_pos_batch(images, pos)
    return images, pos

def get_data_generator(dataloader):
    while True:
        for data in dataloader:
            yield data


def save_model(accelerator: Accelerator, net: torch.nn.Module, optimizer: torch.optim.Optimizer, step, save_path):
    accelerator.wait_for_everyone()
    save_dir = os.path.dirname(save_path)
    if accelerator.is_main_process:
        try:
            os.makedirs(save_dir, exist_ok=True)
            if not os.access(save_dir, os.W_OK):
                print(f"Warning: No write permission for directory {save_dir}")
                return
                
            state_dict = {
                # "model_state_dict": accelerator.get_state_dict(net),
                "model_state_dict": accelerator.unwrap_model(net).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }
            torch.save(state_dict, save_path)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    accelerator.wait_for_everyone()


def is_close(pos, output):
    limit = torch.tensor([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]).to(pos.device)
    # gripper:
    limit[6] = 0.5
    limit[13] = 0.5
    # Handle both single samples and batches
    if pos.dim() == 1:
        return torch.all(torch.abs(pos - output) < limit)
    else:
        return torch.all(torch.abs(pos - output) < limit, dim=1)


def eval(accelerator: Accelerator, net: torch.nn.Module, dataloader: DataLoader, loss_fn, step, use_normalization, mode='val'):
    accelerator.wait_for_everyone()
    net.eval()
    with torch.no_grad():
        eval_loss = 0
        eval_l1_error = 0
        total_correct = 0
        total_samples = 0

        # Get learning dimensions mask from loss function
        learning_mask = loss_fn.learning_mask.to(accelerator.device) if hasattr(loss_fn, 'learning_mask') else torch.ones(14, dtype=torch.bool).to(accelerator.device)
        active_dims = learning_mask.sum().item()
        
        for images, pos in tqdm(dataloader, disable=not accelerator.is_main_process):
            pos = accelerator.gather(pos)
            output = accelerator.gather(net(images))

            if accelerator.is_main_process:
                # Only compute metrics for learned dimensions
                masked_abs_error = torch.abs(pos - output) * learning_mask.float()
                eval_l1_error += (masked_abs_error.sum(dim=1) / active_dims).sum().item() if active_dims > 0 else 0
                
                # For is_close calculation, we only check dimensions we're learning
                if active_dims > 0:
                    is_close_mask = torch.abs(pos - output) < torch.tensor([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.5, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.5]).to(pos.device)
                    # A sample is correct only if all learned dimensions are close
                    correct_samples = is_close_mask[:, learning_mask].float()
                    correct_samples = torch.all(correct_samples, dim=1)
                    total_correct += correct_samples.sum().item()

                total_samples += len(pos)
                
                if mode == 'test':
                    sample_pos = pos[0].detach().cpu().numpy()
                    sample_output = output[0].detach().cpu().numpy()
                    is_correct = is_close(pos[0], output[0]).item()

                    formatted_pos = ', '.join([f"{val:.4f}" for val in sample_pos])
                    formatted_output = ', '.join([f"{val:.4f}" for val in sample_output])
                    
                    print(f"\nSample pos: [{formatted_pos}]")
                    print(f"Sample output: [{formatted_output}]")
                    print(f"Correct?: {is_correct}")
                
                # For loss calculation, normalize pos if normalization is used
                if use_normalization:
                    loss = loss_fn(net.normalize(output), net.normalize(pos))
                else:
                    loss = loss_fn(output, pos)
                eval_loss += loss.item() * len(pos)

        if accelerator.is_main_process:
            eval_loss /= total_samples
            eval_l1_error /= total_samples
            correct_rate = total_correct / total_samples if total_samples > 0 else 0.0
            
            # Print results instead of logging to wandb in eval-only mode
            print(f"{mode}_loss: {eval_loss:.4f}, {mode}_l1_error: {eval_l1_error:.4f}, correct_rate: {correct_rate:.4f}")
            
            # Only log to wandb if it's initialized
            if wandb.run is not None:
                wandb.log({
                    f"{mode}_loss": eval_loss, 
                    f"{mode}_l1_error": eval_l1_error, 
                    f"{mode}_correct_rate": correct_rate
                }, step=step)
    
    net.train()
    accelerator.wait_for_everyone()

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    seed_torch(args.seed)
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    num_gpus = torch.cuda.device_count()
    save_dir = os.path.join(args.save_dir, args.run_name)
    
    # Initialize wandb only if not in eval mode
    if accelerator.is_main_process and not args.eval_only:
        os.makedirs(save_dir, exist_ok=True)
        wandb.init(project=f"IDM_{args.model_name}", mode=args.wandb_mode, config=args.__dict__, name=args.run_name)
    
    if accelerator.is_main_process:
        print(f"{args.__dict__}")

    # Initialize preprocessor
    preprocessor = DinoPreprocessor(args)
    
    # load dataset
    dataset = DataSet(args, dataset_path=args.dataset_path, disable_pbar=not accelerator.is_main_process)
    test_dataset = DataSet(args, dataset_path=args.test_dataset_path, disable_pbar=not accelerator.is_main_process, type="test")
    dataset_size = len(dataset)
    val_dataset_size = min(int(args.ratio_eval * dataset_size), 10000)
    train_dataset_size = dataset_size - val_dataset_size
    train_dataset, val_dataset = random_split(dataset, [train_dataset_size, val_dataset_size])
    if accelerator.is_main_process:
        print('train_dataset_size', train_dataset_size, 'val_dataset_size', val_dataset_size, 'test_dataset_size', len(test_dataset))
    
    use_collate_fn = collate_fn  # or collate_fn_with_split
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, collate_fn=lambda batch: use_collate_fn(batch, preprocessor), drop_last=True, prefetch_factor=args.prefetch_factor)
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=lambda batch: use_collate_fn(batch, preprocessor), drop_last=False, prefetch_factor=args.prefetch_factor)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=lambda batch: use_collate_fn(batch, preprocessor), drop_last=False, prefetch_factor=args.prefetch_factor)

    # Parse learning_dim parameter
    if args.learning_dim is not None:
        args.learning_dim = [int(dim) for dim in args.learning_dim.split(',')]
    else:
        args.learning_dim = list(range(14))  # Default to all dimensions

    if accelerator.is_main_process:
        print(f"Learning dimensions: {args.learning_dim}")

    net = IDM(model_name=args.model_name, dinov2_name=args.dinov2_name, freeze_dinov2=args.freeze_dinov2, output_dim=14)
    
    # Set different learning rates for DINO and DPT models
    # Collect all parameters except dino_model and dpt_model
    base_params = []
    dino_params = []

    # Iterate through all named parameters
    for name, param in net.named_parameters():
        if 'dino_model' in name:
            dino_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {"params": dino_params, "lr": args.learning_rate * 0.1},
        {"params": base_params, "lr": args.learning_rate}
    ]

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    net.train()
    loss_fn = WeightedSmoothL1Loss(beta=0.1, learning_dim=args.learning_dim)

    # Setup learning rate scheduler
    if args.lr_scheduler == "cosine":
        warmup_steps = int(0.1 * args.num_iterations)  # 10% of total steps for warmup
        def lr_lambda(step):
            step = step // num_gpus
            eta_min = 1e-9
            if step < warmup_steps:
                # Linear warmup
                return eta_min + float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, args.num_iterations - warmup_steps))
            return 0.5 * (np.cos(progress * np.pi) + 1)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        if accelerator.is_main_process:
            print(f"Using cosine scheduler with {warmup_steps} warmup steps")
        if accelerator.is_main_process:
            print(f"Using cosine decay scheduler with {warmup_steps} warmup steps")
    else:
        scheduler = None

    if not args.load_from or not os.path.isfile(args.load_from):
        if args.eval_only:
            raise ValueError("Must specify --load_from with a valid model path when using --eval_only")
        start_step = 0
    else:
        # Remove weights_only=True parameter and add error handling
        try:
            loaded_dict = torch.load(args.load_from)
            net.load_state_dict(loaded_dict["model_state_dict"])
            if not args.eval_only:
                optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                start_step = loaded_dict["step"]
                if scheduler is not None:
                    for _ in range(start_step):
                        scheduler.step()
            if accelerator.is_main_process:
                print(f"Loaded model from {args.load_from}")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {args.load_from}: {str(e)}")

    net, optimizer, train_dataloader, val_dataloader, test_dataloader = accelerator.prepare(
        net, optimizer, train_dataloader, val_dataloader, test_dataloader)
    net.normalize = accelerator.unwrap_model(net).normalize
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    if args.eval_only:
        preprocessor.use_transform = False
        eval(accelerator, net, val_dataloader, loss_fn, 0, args.use_normalization, mode='val')
        eval(accelerator, net, test_dataloader, loss_fn, 0, args.use_normalization, mode='test')
        preprocessor.use_transform = args.use_transform
        return

    train_data_generator = get_data_generator(train_dataloader)

    pbar = tqdm(range(start_step, args.num_iterations), disable=not accelerator.is_main_process)
    preprocessor.set_augmentation_progress(0)
    for step in pbar:
        images, pos = next(train_data_generator)
        output = net(images)

        # Calculate batch accuracy using denormalized values
        batch_correct = is_close(pos, output)
        batch_accuracy = batch_correct.float().mean().item()

        if args.use_normalization:
            loss = loss_fn(net.normalize(output), net.normalize(pos))
        else:
            loss = loss_fn(output, pos)
        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if accelerator.is_main_process:
            if isinstance(loss_fn, AdaptiveLoss):
                loss_fn.update_ema()

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            if isinstance(loss_fn, AdaptiveLoss):
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}", ema=f"{loss_fn.ema.cpu().numpy()}", batch_acc=f"{batch_accuracy:.4f}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}", batch_acc=f"{batch_accuracy:.4f}")
            if step % 10 == 0:
                wandb.log({
                    "loss": loss.item(), 
                    "learning_rate": current_lr,
                    "batch_accuracy": batch_accuracy
                }, step=step)

        if (step + 1) % args.eval_interval == 0:
            try:
                preprocessor.use_transform = False
                eval(accelerator, net, val_dataloader, loss_fn, step, args.use_normalization, mode='val')
                eval(accelerator, net, test_dataloader, loss_fn, step, args.use_normalization, mode='test')
                preprocessor.use_transform = args.use_transform
            except Exception as e:
                print(f"Error during evaluation at step {step}: {str(e)}")
            
            save_model(accelerator, net, optimizer, step + 1, os.path.join(save_dir, f"{step + 1}.pt"))
    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    app.run(main(parse_args()))
