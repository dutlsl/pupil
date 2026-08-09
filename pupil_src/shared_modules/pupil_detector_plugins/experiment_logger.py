import os
import datetime
import logging
import torch

logger = logging.getLogger(__name__)

def save_accuracy_log(g_pool, active_model, exp_type, accuracy_value, precision_value=None, rmse_value=None):
    """
    Saves an experiment accuracy log file to recordings directory.
    Uses English names for files and variables.
    """
    try:
        model_name_clean = active_model.replace(" ", "_").replace("(", "").replace(")", "")
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name_clean}_{exp_type}_{now_str}.log"
        
        recordings_dir = os.path.expanduser("~/PycharmProjects/pupil/recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        filepath = os.path.join(recordings_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== Accuracy Log ===\n")
            f.write(f"Model Name: {active_model}\n")
            f.write(f"Experiment Type: {exp_type}\n")
            f.write(f"Execution Time: {datetime.datetime.now().isoformat()}\n")
            
            if rmse_value is not None:
                f.write(f"Calibration RMSE: {rmse_value}\n")
                
            if accuracy_value is not None:
                try:
                    f.write(f"Angular Accuracy: {float(accuracy_value):.3f} degrees\n")
                except ValueError:
                    f.write(f"Angular Accuracy: {accuracy_value}\n")
                    
            if precision_value is not None:
                try:
                    f.write(f"Angular Precision: {float(precision_value):.3f} degrees\n")
                except ValueError:
                    f.write(f"Angular Precision: {precision_value}\n")
                    
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
        logger.info(f"Saved accuracy log to {filepath}")
    except Exception as ex:
        logger.error(f"Failed to save accuracy log: {ex}")
