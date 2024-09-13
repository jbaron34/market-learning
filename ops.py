import torch


def redistribute_negatives(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    n = shape[-1]  # Size of the last dimension
    leading_dims = shape[:-1]  # Leading dimensions

    # Identify negative positions
    neg_mask = (x < 0).float()  # Shape: (*leading_dims, n)

    # Get negative values (as positive numbers)
    neg_values = torch.relu(-x)  # Shape: (*leading_dims, n)

    # Create an identity matrix to exclude self-redistribution
    I = torch.eye(n, dtype=torch.bool, device=x.device)  # Shape: (n, n)

    # Reshape I to be broadcastable to the input tensor shape
    num_leading_dims = len(leading_dims)
    # Reshape I to (1,...,1,n,n) with as many 1s as leading dimensions
    I_shape = (1,) * num_leading_dims + (n, n)
    I = I.view(I_shape)  # Shape: (*1s, n, n)

    # Create a mask to redistribute to all other positions (excluding self)
    redistribution_mask = ~I  # Shape: (*1s, n, n), where ~I is logical NOT

    # Expand neg_mask to match the dimensions for redistribution
    neg_mask_row = neg_mask.unsqueeze(-1)  # Shape: (*leading_dims, n, 1)

    # Create total mask indicating where to redistribute for each negative element
    total_mask = redistribution_mask & (
        neg_mask_row.bool()
    )  # Shape: (*leading_dims, n, n)

    # Number of positions to redistribute to for each negative value
    divisors = ((n - 1) * neg_mask) + (1 - neg_mask)  # Shape: (*leading_dims, n)

    # Compute the redistribution amounts for each position
    neg_values_row = neg_values.unsqueeze(-1)  # Shape: (*leading_dims, n, 1)
    divisors_row = divisors.unsqueeze(-1)  # Shape: (*leading_dims, n, 1)
    redistribution_amounts = total_mask.float() * (
        neg_values_row / divisors_row
    )  # Shape: (*leading_dims, n, n)

    # Sum redistribution amounts for each position across the source dimension
    redistribution = redistribution_amounts.sum(dim=-2)  # Shape: (*leading_dims, n)

    # Set negative values to zero in the original tensor
    x_positives = torch.relu(x)  # Shape: (*leading_dims, n)

    # Add redistributed amounts to all positions
    x_adjusted = x_positives + redistribution  # Shape: (*leading_dims, n)

    return x_adjusted


def select_winning_stakes(
    stakes: torch.Tensor, labels: torch.LongTensor
) -> torch.Tensor:
    class_index = labels.unsqueeze(-1).unsqueeze(-1)
    expanded_class_index = class_index.expand(*stakes.shape[:-1], 1)
    winning_stakes = torch.gather(stakes, -1, expanded_class_index).squeeze(-1)
    return winning_stakes


def calculate_profit(
    stakes: torch.FloatTensor,
    labels: torch.LongTensor,
    pot: int = 1,
    rake: float = 0.2,
) -> torch.FloatTensor:
    pos_stakes = redistribute_negatives(stakes)
    model_totals = pos_stakes.sum(dim=-1)
    pool_totals = (1 - rake) * (pot + model_totals.detach().sum(dim=-1, keepdim=True))
    winning_stakes = select_winning_stakes(pos_stakes, labels)
    winning_proportions = winning_stakes / winning_stakes.detach().sum(
        dim=-1, keepdim=True
    )
    profit = winning_proportions * pool_totals - model_totals
    return -profit.sum(dim=-1).mean()
