

def jaccard_index(predicted: set, real: set) -> float:
    """
    Compute the Jaccard Index (Intersection over Union) between the predicted and real sets of categories.
    
    :param predicted: A set of predicted categories.
    :param real: A set of real (ground truth) categories.
    :return: The Jaccard Index, a float between 0 and 1.
    """
    intersection = len(predicted.intersection(real))
    union = len(predicted.union(real))
    
    if union == 0:
        return 1.0 if len(real) == 0 else 0.0  # Handle edge case when both are empty
    
    return intersection / union

def precision(predicted: set, real: set) -> float:
    """
    Compute precision as the proportion of predicted categories that are correct.
    
    :param predicted: A set of predicted categories.
    :param real: A set of real (ground truth) categories.
    :return: Precision score, a float between 0 and 1.
    """
    if len(predicted) == 0:
        return 1.0 if len(real) == 0 else 0.0  # Handle edge case when both are empty
    
    intersection = len(predicted.intersection(real))
    return intersection / len(predicted)

def recall(predicted: set, real: set) -> float:
    """
    Compute recall as the proportion of real categories that were correctly predicted.
    
    :param predicted: A set of predicted categories.
    :param real: A set of real (ground truth) categories.
    :return: Recall score, a float between 0 and 1.
    """
    if len(real) == 0:
        return 1.0 if len(predicted) == 0 else 0.0  # Handle edge case when both are empty
    
    intersection = len(predicted.intersection(real))
    return intersection / len(real)

def f1_score(predicted: set, real: set) -> float:
    """
    Compute the F1 score, which is the harmonic mean of precision and recall.
    
    :param predicted: A set of predicted categories.
    :param real: A set of real (ground truth) categories.
    :return: F1 score, a float between 0 and 1.
    """
    prec = precision(predicted, real)
    rec = recall(predicted, real)
    
    if prec + rec == 0:
        return 0.0  # Handle edge case when both precision and recall are zero
    
    return 2 * (prec * rec) / (prec + rec)


# # Example of usage
# predicted_categories = {"cat1", "cat2"}
# real_categories = {"cat2", "cat3", "cat1"}

# print("Jaccard Index:", jaccard_index(predicted_categories, real_categories))
# print("Precision:", precision(predicted_categories, real_categories))
# print("Recall:", recall(predicted_categories, real_categories))
# print("F1 Score:", f1_score(predicted_categories, real_categories))