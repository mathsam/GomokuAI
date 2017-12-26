def count_likely_moves(target_move_prob, pred_move_prob, top_k=4):
    """Count how many top moves are correctly predicted"""
    num_match = 0
    num_samples = target_move_prob.shape[0]
    for i in range(num_samples):
        target_best = set(target_move_prob[i, :].argsort()[-top_k:])
        pred_best = set(pred_move_prob[i, :].argsort()[-top_k:])
        num_match += len(target_best.intersection(pred_best))
    return num_match / float(top_k * num_samples)
