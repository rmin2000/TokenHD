import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class WeightedVectorOptimizer:
    def __init__(self, num_models, learning_rate=0.01, device="cpu", weighted_loss=False):
        self.device = device
        self.learning_rate = learning_rate
        self.num_models = num_models
        self.weighted_loss = weighted_loss

        equal_weight = 1.0 / num_models
        self.raw_params = nn.Parameter(
            torch.full((num_models - 1,), equal_weight, device=device, requires_grad=True)
        )
        print("Initial weights:", self.get_normalized_params())
        self.optimizer = optim.Adam([self.raw_params], lr=learning_rate)
        self.scheduler = None
        self.criterion = nn.MSELoss()

    def get_normalized_params(self):
        first_params = self.raw_params
        last_param = 1.0 - torch.sum(first_params)
        all_params = torch.cat([last_param.unsqueeze(0), first_params])
        return all_params

    def forward(self, input_vectors_list):
        params = self.get_normalized_params()
        weighted_sum = torch.zeros_like(input_vectors_list[0])
        for i, vectors in enumerate(input_vectors_list):
            weighted_sum += params[i] * vectors
        return weighted_sum

    def compute_loss(self, predictions, gt_vectors):
        mse_loss = (predictions - gt_vectors) ** 2

        gt_high_mask = (gt_vectors > 0.5).float()
        gt_low_mask = (gt_vectors <= 0.5).float()
        total_tokens = gt_vectors.numel()
        high_token_count = torch.sum(gt_high_mask)
        low_token_count = torch.sum(gt_low_mask)

        weight_for_high = low_token_count / total_tokens if total_tokens > 0 else 0.0
        weight_for_low = high_token_count / total_tokens if total_tokens > 0 else 0.0

        weighted_loss = gt_high_mask * weight_for_high * mse_loss + gt_low_mask * weight_for_low * mse_loss

        if not self.weighted_loss:
            return torch.mean(mse_loss)
        else:
            return torch.mean(weighted_loss)

    def train(self, input_vectors_list, gt_vectors, epochs=100, batch_size=64):
        best_loss = float("inf")
        best_params = None
        loss_history = []

        total_samples = len(gt_vectors)
        num_batches = (total_samples + batch_size - 1) // batch_size

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0.0
            batch_count = 0
            sample_indices = torch.randperm(total_samples).tolist()

            for batch_idx in range(num_batches):
                batch_loss = 0.0
                batch_sample_count = 0
                self.optimizer.zero_grad()

                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                batch_sample_indices = sample_indices[start_idx:end_idx]

                for sample_idx in batch_sample_indices:
                    sample_gt = torch.tensor(gt_vectors[sample_idx], dtype=torch.float32, device=self.device)
                    sample_preds = [
                        torch.tensor(input_vectors_list[m][sample_idx], dtype=torch.float32, device=self.device)
                        for m in range(len(input_vectors_list))
                    ]

                    predictions = self.forward(sample_preds)
                    loss = self.compute_loss(predictions, sample_gt)
                    loss.backward()
                    batch_loss += loss.item()
                    batch_sample_count += 1

                self.optimizer.step()
                if batch_sample_count > 0:
                    epoch_loss += batch_loss
                    batch_count += batch_sample_count

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")
            loss_history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = self.get_normalized_params().detach().cpu().numpy().copy()

        print(f"Final parameters: {best_params}")
        print(f"Final loss: {best_loss:.6f}")
        return best_params, None
