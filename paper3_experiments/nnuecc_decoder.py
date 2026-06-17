def decode_halfkpa_to_planes(white_indices, white_values, black_indices):
    """
    white_indices: [B, N] tensor
    white_values: [B, N] tensor (1.0 for valid, 0.0 for padding)
    black_indices: [B, N] tensor
    """
    B, N = white_indices.shape
    device = white_indices.device
    
    planes = torch.zeros(B, 12, 64, device=device, dtype=torch.float32)
    
    # 1. Parse white_indices for the 10 non-king pieces
    valid_mask = (white_values > 0).float()
    
    sq = white_indices % 64
    p_idx = (white_indices // 64) % 10
    
    # Mapping p_idx to PMP plane index
    # PMP planes: 
    # 0:WP, 1:WN, 2:WB, 3:WR, 4:WQ, 5:WK
    # 6:BP, 7:BN, 8:BB, 9:BR, 10:BQ, 11:BK
    # Stockfish p_idx:
    # 0:WP, 1:BP, 2:WN, 3:BN, 4:WB, 5:BB, 6:WR, 7:BR, 8:WQ, 9:BQ
    mapping = torch.tensor([0, 6, 1, 7, 2, 8, 3, 9, 4, 10], device=device, dtype=torch.long)
    plane_idx = mapping[p_idx]
    
    # Batch indices
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    
    # Scatter the pieces
    # We add values, using valid_mask to zero out padded elements
    # Since pieces are at most 1 per square per piece type, we can just sum
    planes.view(B, -1).put_(
        batch_idx * (12 * 64) + plane_idx * 64 + sq,
        valid_mask,
        accumulate=True
    )
    
    # 2. Parse White King
    # The King square is stationary per batch element, so we just take the first valid index
    # Actually, all elements in a row have the same king_sq, even padding if it copies the king_sq.
    # We can just take the max to be safe, or just index 0 (assuming N > 0 and index 0 is valid)
    wk_sq = white_indices[:, 0] // 640
    batch_idx_1d = torch.arange(B, device=device)
    planes[batch_idx_1d, 5, wk_sq] = 1.0
    
    # 3. Parse Black King
    # bk_sq_mirrored = black_indices[:, 0] // 640
    # bk_sq = bk_sq_mirrored ^ 56
    bk_sq = (black_indices[:, 0] // 640) ^ 56
    planes[batch_idx_1d, 11, bk_sq] = 1.0
    
    # Reshape to [B, 12, 8, 8]
    return planes.view(B, 12, 8, 8)

