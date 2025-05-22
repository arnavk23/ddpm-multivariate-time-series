def log_metrics(metrics, step):
    """
    Logs the given metrics at the specified step.
    
    Parameters:
    metrics (dict): A dictionary containing metric names and their values.
    step (int): The current step or epoch number.
    """
    for key, value in metrics.items():
        print(f"Step {step}: {key} = {value}")


def save_results(results, filename):
    """
    Saves the results to a specified file.
    
    Parameters:
    results (dict): A dictionary containing results to be saved.
    filename (str): The name of the file where results will be saved.
    """
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
