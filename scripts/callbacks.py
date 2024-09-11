import time
import mlflow
import psutil
import torch
from transformers import TrainerCallback
from args import simple_train_arguments , fsdp_train_arguments


train_arguments = fsdp_train_arguments

class HardwareUtilizationCallback(TrainerCallback):
    def __init__(self, logging_steps=100):
        super().__init__()
        self.logging_steps = logging_steps
        self.num_gpus = torch.cuda.device_count()
        self.global_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """
        This method is called at the end of each training step.
        """
        self.global_step += 1
        if self.global_step % self.logging_steps == 0:
            # CPU Utilization
            cpu_utilization = psutil.cpu_percent(interval=None)
            mlflow.log_metric("cpu_utilization", cpu_utilization, step=self.global_step)
            
            # Memory Utilization
            virtual_memory = psutil.virtual_memory()
            mlflow.log_metric("ram_usage", virtual_memory.used / (1024 ** 3), step=self.global_step)  # Convert to GB
            mlflow.log_metric("ram_available", virtual_memory.available / (1024 ** 3), step=self.global_step)  # Convert to GB

            # GPU Utilization and Memory Usage
            for gpu_index in range(self.num_gpus):
                if torch.cuda.is_available():
                    gpu_utilization = torch.cuda.utilization(gpu_index)
                    gpu_memory_allocated = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)  # Convert to GB
                    gpu_memory_reserved = torch.cuda.memory_reserved(gpu_index) / (1024 ** 3)  # Convert to GB
                
                    mlflow.log_metric(f"gpu_{gpu_index}_utilization", gpu_utilization, step=self.global_step)
                    mlflow.log_metric(f"gpu_{gpu_index}_memory_allocated", gpu_memory_allocated, step=self.global_step)
                    mlflow.log_metric(f"gpu_{gpu_index}_memory_reserved", gpu_memory_reserved, step=self.global_step)

                    print(f"Step {self.global_step}: GPU {gpu_index} Utilization {gpu_utilization}%, Memory Allocated {gpu_memory_allocated}GB")

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            mlflow.log_metric("disk_read_bytes", disk_io.read_bytes / (1024 ** 3), step=self.global_step)  # Convert to GB
            mlflow.log_metric("disk_write_bytes", disk_io.write_bytes / (1024 ** 3), step=self.global_step)  # Convert to GB

            # Print for debugging (optional)
            print(f"Step {self.global_step}: CPU {cpu_utilization}%, RAM {virtual_memory.used / (1024 ** 3)}GB")



class SpeedMonitoringCallback(TrainerCallback):
    def __init__(self, logging_steps):
        self.logging_steps = logging_steps
        self.total_tokens_processed = 0
        self.steps = 0
        self.epoch_start_time = None
        self.step_start_time = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("Training started.")

    def calculate_and_log_metrics(self, args, state, elapsed_time, tokens_processed, step_or_epoch_label):
        throughput = self.total_tokens_processed / (time.time() - self.start_time)
        mlflow.log_metric(f"{step_or_epoch_label}_throughput", throughput, step=state.global_step)
        mlflow.log_metric(f"{step_or_epoch_label}_elapsed_time", elapsed_time, step=state.global_step)
        print(f"{step_or_epoch_label.capitalize()} {state.global_step} | Throughput: {throughput:.2f} tokens/s")

    def on_step_begin(self, args, state, control, **kwargs):
        if self.steps % self.logging_steps == 0:
            self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        self.steps += 1
        if self.steps % self.logging_steps == 0:
            elapsed_time = time.time() - self.step_start_time
            
            # Calculate the number of tokens processed
            tokens_per_batch = (
                args.per_device_train_batch_size 
                * train_arguments().max_seq_length
                * args.gradient_accumulation_steps  # Adjust for gradient accumulation
            )
            
            # If running distributed training, multiply by world size (number of devices)
            world_size = args.world_size if hasattr(args, 'world_size') else 1
            tokens_processed = tokens_per_batch * self.logging_steps * world_size
            
            self.total_tokens_processed += tokens_processed
            self.calculate_and_log_metrics(args, state, elapsed_time, tokens_processed, "step")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"Epoch {state.epoch} begins.")

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed_time = time.time() - self.epoch_start_time
        
        # Calculate total tokens processed in the entire epoch
        tokens_per_batch = (
            args.per_device_train_batch_size 
            * train_arguments().max_seq_length
            * args.gradient_accumulation_steps  # Adjust for gradient accumulation
        )
        
        # If running distributed training, multiply by world size (number of devices)
        world_size = args.world_size if hasattr(args, 'world_size') else 1
        tokens_processed = tokens_per_batch * self.steps * world_size
        
        self.calculate_and_log_metrics(args, state, elapsed_time, tokens_processed, "epoch")
        print(f"Epoch {state.epoch} ended. Duration: {elapsed_time:.2f}s")


