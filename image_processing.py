# image_processing.py
"""
Image Processing Pipeline

This module defines a set of image processing steps (like brightness, blur, crop, etc.)
and lets you chain them together in a pipeline. Each step takes an image and outputs a modified version.
"""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import json

# Dataclass to define parameters for each image processing step
@dataclass
class StepParameter:
    """Defines a parameter for an image processing step"""
    name: str
    type: type
    description: str
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

# Abstract base class for all image processing steps
class ImageProcessingStep(ABC):
    
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the image with the given parameters"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[StepParameter]:
        """Return the parameters required by this step"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this step"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this step does"""
        pass


class BrightnessStep(ImageProcessingStep):
    """Adjust image brightness"""
    
    @property
    def name(self) -> str:
        return "brightness"
    
    @property
    def description(self) -> str:
        return "Adjust the brightness of the image"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="factor",
                type=float,
                description="Brightness factor (0.0 = black, 1.0 = original, 2.0 = twice as bright)",
                default=1.0,
                min_value=0.0,
                max_value=3.0
            )
        ]
    
    def process(self, image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Adjust brightness by multiplying pixel values"""
        result = image.astype(np.float32) * factor
        return np.clip(result, 0, 255).astype(np.uint8)


class SaturationStep(ImageProcessingStep):
    """Adjust image saturation"""
    
    @property
    def name(self) -> str:
        return "saturation"
    
    @property
    def description(self) -> str:
        return "Adjust the color saturation of the image"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="factor",
                type=float,
                description="Saturation factor (0.0 = grayscale, 1.0 = original, 2.0 = double saturation)",
                default=1.0,
                min_value=0.0,
                max_value=3.0
            )
        ]
    
    def process(self, image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Adjust saturation in HSV color space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * factor   # Scale saturation channel
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class HueStep(ImageProcessingStep):
    """Adjust image hue"""
    
    @property
    def name(self) -> str:
        return "hue"
    
    @property
    def description(self) -> str:
        return "Shift the hue (color) of the image"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="shift",
                type=int,
                description="Hue shift in degrees (-180 to 180)",
                default=0,
                min_value=-180,
                max_value=180
            )
        ]
    
    def process(self, image: np.ndarray, shift: int = 0) -> np.ndarray:
        """Shift hue in HSV color space."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift/2) % 180  # OpenCV hue range
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class BoxBlurStep(ImageProcessingStep):
    """Apply box blur filter (custom implementation)"""
    
    @property
    def name(self) -> str:
        return "box_blur"
    
    @property
    def description(self) -> str:
        return "Apply a box blur filter to the image"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="kernel_size",
                type=int,
                description="Size of the blur kernel (must be odd)",
                default=5,
                min_value=3,
                max_value=31
            )
        ]
    
    def process(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Applying box blur using optimized vectorized implementation for optimized performance"""
        if kernel_size % 2 == 0:
            kernel_size += 1  # Odd size only as kernel needs to have a center

        
        # For custom implementation, I am using a fast vectorized version

        # implementing using cumulative sums
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis] #Ensure 3D shape
            
        h, w, c = image.shape
        output = np.zeros_like(image, dtype=np.float32)
        
        # Pad size
        pad = kernel_size // 2
        
        for channel in range(c):
            # Getting channel and padding it
            img = image[:, :, channel].astype(np.float32)
            padded = np.pad(img, pad, mode='edge')
            
            # Building integral image to speed up kernel averaging
            cumsum = np.zeros((h + 2*pad + 1, w + 2*pad + 1), dtype=np.float32)
            cumsum[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)
            
            # Vectorized box sum calculation
            # Calculating all sums at once using slicing
            y1 = np.arange(h)
            x1 = np.arange(w)
            y2 = y1 + kernel_size
            x2 = x1 + kernel_size
            
            # I am using broadcasting to compute all sums at once
            Y1, X1 = np.meshgrid(y1, x1, indexing='ij')
            Y2, X2 = np.meshgrid(y2, x2, indexing='ij')
            
            sums = (cumsum[Y2, X2] - cumsum[Y1, X2] - 
                   cumsum[Y2, X1] + cumsum[Y1, X1])
            
            output[:, :, channel] = sums / (kernel_size * kernel_size)
        
        if c == 1:
            output = output[:, :, 0]
            
        return np.clip(output, 0, 255).astype(np.uint8)


class UnsharpMaskStep(ImageProcessingStep):
    """Apply an unsharp mask to enhance image sharpness"""
    
    def __init__(self, blur_step: BoxBlurStep):
        self.blur_step = blur_step  #Reusing our own box blur implementation
    
    @property
    def name(self) -> str:
        return "unsharp_mask"
    
    @property
    def description(self) -> str:
        return "Sharpen the image using unsharp mask technique"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="strength",
                type=float,
                description="Sharpening strength",
                default=1.0,
                min_value=0.0,
                max_value=5.0
            ),
            StepParameter(
                name="blur_size",
                type=int,
                description="Size of blur kernel for unsharp mask",
                default=5,
                min_value=3,
                max_value=21
            )
        ]
    
    def process(self, image: np.ndarray, strength: float = 1.0, blur_size: int = 5) -> np.ndarray:
        """Apply unsharp mask: original + strength * (original - blurred)"""
        blurred = self.blur_step.process(image, kernel_size=blur_size)
        
        # Calculate the difference
        diff = image.astype(np.float32) - blurred.astype(np.float32)
        
        # Applying unsharp mask
        sharpened = image.astype(np.float32) + strength * diff
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)

#Cropping is useful for isolating regions of interest or cutting out artifacts
class CropStep(ImageProcessingStep):
    """Crop the image to specified region"""
    
    @property
    def name(self) -> str:
        return "crop"
    
    @property
    def description(self) -> str:
        return "Crop the image to a specified region"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="x",
                type=int,
                description="X coordinate of top-left corner",
                default=0,
                min_value=0
            ),
            StepParameter(
                name="y",
                type=int,
                description="Y coordinate of top-left corner",
                default=0,
                min_value=0
            ),
            StepParameter(
                name="width",
                type=int,
                description="Width of crop region",
                default=100,
                min_value=1
            ),
            StepParameter(
                name="height",
                type=int,
                description="Height of crop region",
                default=100,
                min_value=1
            )
        ]
    
    def process(self, image: np.ndarray, x: int = 0, y: int = 0, 
                width: int = 100, height: int = 100) -> np.ndarray:
        """Crop image to specified region"""
        h, w = image.shape[:2]
        
        # Ensuring crop region is within bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        x2 = min(x + width, w)
        y2 = min(y + height, h)
        
        return image[y:y2, x:x2].copy() # Return cropped region


class RotateStep(ImageProcessingStep):
    """Rotate image in 90-degree increments"""
    
    @property
    def name(self) -> str:
        return "rotate"
    
    @property
    def description(self) -> str:
        return "Rotate the image in 90-degree increments"
    
    def get_parameters(self) -> List[StepParameter]:
        return [
            StepParameter(
                name="angle",
                type=int,
                description="Rotation angle (must be 0, 90, 180, or 270)",
                default=0
            )
        ]
    
    def process(self, image: np.ndarray, angle: int = 0) -> np.ndarray:
        """Rotate image by specified angle (0, 90, 180, or 270 degrees)"""
        angle = angle % 360
        
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # For non-90 degree increments, round to nearest 90
            nearest_90 = round(angle / 90) * 90
            return self.process(image, nearest_90)

# Pipeline class to orchestrate all the steps in a user-defined sequence
class ImageProcessingPipeline:
    """Main pipeline class that composes and executes image processing steps"""
    
    def __init__(self):
        """Initialize pipeline with all available steps."""
        blur_step = BoxBlurStep()
        
        self.available_steps = {
            "brightness": BrightnessStep(),
            "saturation": SaturationStep(),
            "hue": HueStep(),
            "box_blur": blur_step,
            "unsharp_mask": UnsharpMaskStep(blur_step),
            "crop": CropStep(),
            "rotate": RotateStep()
        }
    
    def get_available_steps(self) -> Dict[str, Dict[str, Any]]:
        """Return information about all available steps and their parameters"""
        steps_info = {}
        
        for step_name, step in self.available_steps.items():
            params_info = []
            for param in step.get_parameters():
                param_info = {
                    "name": param.name,
                    "type": param.type.__name__,
                    "description": param.description,
                    "default": param.default
                }
                if param.min_value is not None:
                    param_info["min_value"] = param.min_value
                if param.max_value is not None:
                    param_info["max_value"] = param.max_value
                params_info.append(param_info)
            
            steps_info[step_name] = {
                "name": step.name,
                "description": step.description,
                "parameters": params_info
            }
        
        return steps_info
    
    def process_image(self, image: np.ndarray, pipeline: List[Dict[str, Any]]) -> np.ndarray:
        """
        Process an image through a pipeline of steps.
        
        Args:
            image: Input image as numpy array
            pipeline: List of dicts, each containing 'step' name and 'params' dict
            
        Returns:
            Processed image as numpy array
        """
        # Work on a copy to avoid modifying the original
        result = image.copy()
        
        # Ensuring image is contiguous in memory for better cache performance
        if not result.flags['C_CONTIGUOUS']:
            result = np.ascontiguousarray(result)
        
        for step_config in pipeline:
            step_name = step_config.get('step')
            params = step_config.get('params', {})
            
            if step_name not in self.available_steps:
                raise ValueError(f"Unknown step: {step_name}")
            
            step = self.available_steps[step_name]
            result = step.process(result, **params)
            
            # Making sure result is still contiguous after processing
            if not result.flags['C_CONTIGUOUS']:
                result = np.ascontiguousarray(result)
        
        return result
    
    def process_image_from_file(self, image_path: str, pipeline: List[Dict[str, Any]], 
                               output_path: Optional[str] = None) -> np.ndarray:
        """
        Convenience method to process an image file
        
        Args:
            image_path: Path to input image
            pipeline: Pipeline configuration
            output_path: Optional path to save the result
            
        Returns:
            Processed image as numpy array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}") # Likely bad path or format
        
        result = self.process_image(image, pipeline)
        
        if output_path:
            cv2.imwrite(output_path, result)
        
        return result


# Pipeline class to orchestrate all the steps in a user-defined sequence
if __name__ == "__main__":
    # Creating pipeline instance
    pipeline = ImageProcessingPipeline()
    
    # Getting all the available steps
    steps = pipeline.get_available_steps()
    print("Available steps:")
    print(json.dumps(steps, indent=2))
    
    # Testing pipeline configuration before implementing UI
    testing_pipeline = [
        {"step": "brightness", "params": {"factor": 1.2}},
        {"step": "saturation", "params": {"factor": 1.5}},
        {"step": "box_blur", "params": {"kernel_size": 7}},
        {"step": "unsharp_mask", "params": {"strength": 1.5, "blur_size": 5}}
    ]
    
    # Process any image (make sure xyz.png is in the root folder)
    try:
        result = pipeline.process_image_from_file(
            "jumbo.jpg", 
            testing_pipeline, 
            "output.jpg"
        )
        print("\nImage processed successfully!")
    except Exception as e:
        print(f"\nNote: {e}")
        print("Please provide a test image to process.")