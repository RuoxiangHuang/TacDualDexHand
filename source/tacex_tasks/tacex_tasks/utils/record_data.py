"""Base class for recording data in IsaacLab environments."""

import os
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


def get_unique_filename(base_path: str, extension: str = ".npz") -> str:
    """Generate a unique filename by appending a number if the file exists.
    
    Args:
        base_path: Base path without extension (e.g., "Data/task_name/train_data/data")
        extension: File extension (e.g., ".npz")
    
    Returns:
        Unique filename with extension
    """
    if not extension.startswith("."):
        extension = "." + extension
    
    file_path = base_path + extension
    counter = 0
    
    while os.path.exists(file_path):
        counter += 1
        file_path = f"{base_path}_{counter}{extension}"
    
    return file_path


class RecordDataEnv(ABC):
    """Base class for recording data in IsaacLab environments.
    
    This class provides functionality for recording data during simulation and saving it to npz files.
    It is designed to be used with IsaacLab environments by inheriting from this class and
    implementing the `record_callback` method.
    
    Usage:
        1. Inherit from RecordDataEnv in your IsaacLab environment class
        2. Override the `record_callback` method to collect data
        3. Call `self._record_step()` in your `_pre_physics_step` method
        4. Call `self.start_record(task_name)` to begin recording
        5. Call `self.stop_record()` to stop recording
        6. Call `self.record_to_npz()` to save data to npz file
    """
    
    def __init__(self):
        """Initialize the RecordDataEnv."""
        # Recording flag
        self.record_flag = False
        
        # Save recording data
        self.saving_data = []
        
        # Data dictionary for saving to npz
        self.saving_data_replay = {}
        
        # Step counter
        self.step_num = 0
        
        # Task name for saving
        self.record_task_name: Optional[str] = None
        
        # Data split index (for separating training and remaining data)
        self.data_split: Optional[int] = None
    
    def start_record(self, task_name: str, data_dir: str = "Data"):
        """Start recording data.
        
        Args:
            task_name: Name of the task (used for directory naming)
            data_dir: Base directory for saving data (default: "Data")
        """
        if not self.record_flag:
            self.record_flag = True
            self.step_num = 0
            
            # Create directory if it doesn't exist
            save_dir = os.path.join(data_dir, task_name, "train_data")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            self.record_task_name = task_name
            self.data_dir = data_dir
    
    def stop_record(self):
        """Stop recording and prepare data for saving."""
        if self.record_flag:
            self.record_flag = False
            
            # Prepare data for saving
            if len(self.saving_data) > 0:
                self.saving_data_replay["stage_all"] = np.array(self.saving_data)
                
                if self.data_split is not None:
                    self.saving_data_replay["stage_remain"] = np.array(
                        self.saving_data[self.data_split:]
                    )
                    print(f"Data length: {len(self.saving_data)}")
                    print(f"Data split index: {self.data_split}")
                    print(f"Remain Data length: {len(self.saving_data[self.data_split:])}")
                else:
                    print(f"Data length: {len(self.saving_data)}")
            
            # Clear record data (but keep saving_data_replay for record_to_npz)
            self.saving_data = []
    
    def record_to_npz(self, **kwargs) -> str:
        """Save recorded data to npz file.
        
        Args:
            **kwargs: Additional data to add to saving_data_replay before saving
        
        Returns:
            Path to the saved npz file
        """
        # Add any additional data provided via kwargs
        self.saving_data_replay.update(kwargs)
        
        # Generate unique filename
        base_path = os.path.join(
            self.data_dir, 
            self.record_task_name, 
            "train_data", 
            "data"
        )
        record_file_name = get_unique_filename(base_path, ".npz")
        
        # Save to npz
        np.savez_compressed(record_file_name, **self.saving_data_replay)
        print(f"Record data saved to {record_file_name}")
        
        return record_file_name
    
    def _record_step(self):
        """Internal method to be called in _pre_physics_step.
        
        This method should be called in the _pre_physics_step method of your environment.
        It will call record_callback if recording is enabled.
        """
        if self.record_flag:
            # Call the user-defined record_callback
            self.record_callback()
            self.step_num += 1
    
    @abstractmethod
    def record_callback(self):
        """Record callback to be implemented by subclasses.
        
        This method should be overridden in subclasses to collect the desired data.
        The collected data should be appended to self.saving_data as a dictionary.
        
        Example:
            def record_callback(self):
                if self.step_num % 5 == 0:
                    data = {
                        "joint_state": self.robot.data.joint_pos.cpu().numpy(),
                        "ee_pos": self.robot.data.ee_pos_w.cpu().numpy(),
                    }
                    self.saving_data.append(data)
        """
        pass
