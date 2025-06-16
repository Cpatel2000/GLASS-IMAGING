# app.py
"""
FastAPI Web Application for Image Processing Pipeline

This module provides a web interface for composing and executing
image processing pipelines on uploaded images.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import base64
from typing import List, Dict, Any
import io
from image_processing import ImageProcessingPipeline
import json
import os

app = FastAPI(title="Image Processing Pipeline")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the image processing pipeline
pipeline = ImageProcessingPipeline()


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array image to base64 string with optimization."""
    # Use JPEG for better compression and faster encoding
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"


def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array image."""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML page."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Processing Pipeline</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: 
                radial-gradient(ellipse at top left, rgba(30, 58, 138, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at top right, rgba(91, 33, 182, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at bottom left, rgba(37, 99, 235, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(109, 40, 217, 0.1) 0%, transparent 50%),
                linear-gradient(180deg, 
                    #050510 0%, 
                    #0a0a1f 20%, 
                    #0f0f2e 40%, 
                    #151535 60%, 
                    #1a1a4e 80%, 
                    #1e1b4b 100%
                );
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Darkened animated vertical stripes overlay */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: -100%;
            width: 200%;
            height: 100%;
            background: repeating-linear-gradient(
                90deg,
                transparent 0px,
                rgba(30, 58, 138, 0.02) 2px,
                rgba(37, 99, 235, 0.03) 3px,
                rgba(55, 48, 163, 0.02) 4px,
                transparent 5px,
                transparent 10px,
                rgba(79, 70, 229, 0.02) 11px,
                rgba(91, 33, 182, 0.03) 12px,
                rgba(109, 40, 217, 0.02) 13px,
                transparent 14px,
                transparent 20px
            );
            animation: stripeMove 30s linear infinite;
            pointer-events: none;
            z-index: 1;
        }
        
        /* Darkened light effect overlay */
        body::after {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: 
                radial-gradient(circle at 20% 50%, rgba(30, 58, 138, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(91, 33, 182, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(55, 48, 163, 0.03) 0%, transparent 70%);
            animation: lightMove 20s ease-in-out infinite;
            pointer-events: none;
            z-index: 1;
            opacity: 0.5;
        }
        
        @keyframes lightMove {
            0%, 100% {
                transform: rotate(0deg) scale(1);
                opacity: 0.5;
            }
            50% {
                transform: rotate(180deg) scale(1.1);
                opacity: 0.8;
            }
        }
        
        /* Container to ensure content is above effects */
        .container {
            position: relative;
            z-index: 2;
        }
        
        @keyframes stripeMove {
            0% {
                transform: translateX(0);
            }
            100% {
                transform: translateX(50%);
            }
        }
        
        .glass-card {
            background: linear-gradient(135deg, 
                rgba(17, 24, 39, 0.7) 0%, 
                rgba(31, 41, 55, 0.5) 50%, 
                rgba(55, 65, 81, 0.3) 100%
            );
            backdrop-filter: blur(20px) saturate(1.2);
            border: 1px solid rgba(75, 85, 99, 0.2);
            box-shadow: 
                0 8px 32px 0 rgba(0, 0, 0, 0.6),
                inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        /* Subtle stripe pattern on cards */
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                90deg,
                transparent 0px,
                rgba(156, 163, 175, 0.02) 2px,
                transparent 4px
            );
            pointer-events: none;
        }
        
        .glass-card-light {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        }
        
        @keyframes gradient-x {
            0%, 100% {
                background-size: 200% 200%;
                background-position: left center;
            }
            50% {
                background-size: 200% 200%;
                background-position: right center;
            }
        }
        
        .animate-gradient-x {
            animation: gradient-x 3s ease infinite;
        }
        
        .step-card {
            background: linear-gradient(135deg, 
                rgba(17, 24, 39, 0.8) 0%, 
                rgba(31, 41, 55, 0.6) 50%, 
                rgba(55, 65, 81, 0.4) 100%
            );
            backdrop-filter: blur(10px) saturate(1.2);
            border: 1px solid rgba(75, 85, 99, 0.3);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            border-radius: 20px;
            box-shadow: 
                0 10px 30px -10px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        /* Gradient border effect */
        .step-card::after {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, 
                #1e3a8a, #312e81, #4c1d95, #5b21b6, #6d28d9, #4c1d95, #312e81, #1e3a8a
            );
            background-size: 300% 300%;
            border-radius: 20px;
            opacity: 0;
            z-index: -1;
            transition: opacity 0.3s ease;
            animation: gradientShift 4s ease infinite;
        }
        
        .step-card:hover {
            transform: translateY(-5px) scale(1.02);
            background: linear-gradient(135deg, 
                rgba(31, 41, 55, 0.9) 0%, 
                rgba(55, 65, 81, 0.7) 50%, 
                rgba(75, 85, 99, 0.5) 100%
            );
            box-shadow: 
                0 20px 40px -15px rgba(0, 0, 0, 0.7),
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 0 30px rgba(55, 48, 163, 0.2);
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Glow effect on hover */
        .step-card:hover::after {
            opacity: 0.8;
        }
        
        /* Inner glow */
        .step-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(255, 255, 255, 0.1) 50%,
                transparent 100%
            );
            transition: left 0.5s ease;
            pointer-events: none;
        }
        
        .step-card:hover::before {
            left: 100%;
        }
        
        .step-card:hover {
            transform: translateY(-5px) scale(1.02);
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
            box-shadow: 
                0 20px 40px -15px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 0 30px rgba(102, 126, 234, 0.3);
        }
        
        /* Icon container */
        .step-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 15px;
            font-size: 24px;
            filter: grayscale(100%);
            opacity: 0.8;
            transition: all 0.3s ease;
            box-shadow: 
                0 5px 15px -5px rgba(0, 0, 0, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
        }
        
        .step-card:hover .step-icon {
            opacity: 1;
            transform: scale(1.1);
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
        }
        
        /* Fancy parameter count badge */
        .param-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            box-shadow: 0 3px 10px -3px rgba(245, 87, 108, 0.5);
        }
        
        .pipeline-item {
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(5px);
            border: 1px solid rgba(55, 65, 81, 0.4);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .pipeline-item::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: repeating-linear-gradient(
                90deg,
                transparent 0px,
                rgba(75, 85, 99, 0.05) 1px,
                transparent 3px
            );
            pointer-events: none;
        }
        
        .image-container {
            max-height: 600px;
            overflow: hidden;
            background: rgba(0, 0, 0, 0.6);
            position: relative;
            border-radius: 12px;
            border: 1px solid rgba(55, 65, 81, 0.3);
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 600px;
            object-fit: contain;
            position: relative;
            z-index: 0;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 14px;
            border: none;
            box-shadow: 0 4px 15px -3px rgba(102, 126, 234, 0.4);
        }
        
        .btn-primary::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .btn-primary:hover::before {
            left: 100%;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(102, 126, 234, 0.5);
        }
        
        .btn-danger {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 14px;
            border: none;
            box-shadow: 0 4px 15px -3px rgba(245, 87, 108, 0.4);
        }
        
        .btn-danger::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }
        
        .btn-danger:hover::before {
            left: 100%;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(245, 87, 108, 0.5);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body class="text-white">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-5xl font-bold text-center mb-8 text-gray-300">
            Image Processing Pipeline
        </h1>
        
        <!-- File Upload Section -->
        <div class="glass-card rounded-2xl p-8 mb-8">
            <h2 class="text-2xl font-bold mb-6 text-white flex items-center">
                <svg class="w-8 h-8 mr-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                </svg>
                Upload Image
            </h2>
            <div class="relative">
                <input type="file" id="imageInput" accept="image/*" class="hidden">
                <label for="imageInput" class="block w-full cursor-pointer">
                    <div class="border-2 border-dashed border-gray-700 rounded-2xl p-12 text-center hover:border-gray-600 transition-all duration-300 hover:bg-white/5">
                        <div class="mb-4">
                            <svg class="mx-auto h-16 w-16 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                        </div>
                        <p class="text-lg font-medium text-gray-400 mb-2">Drop your image here</p>
                        <p class="text-sm text-gray-500">or click to browse</p>
                        <p class="text-xs text-gray-600 mt-2">Supports: JPG, PNG, GIF (Max 10MB)</p>
                    </div>
                </label>
            </div>
        </div>
        
        <!-- Available Steps Section -->
        <div class="glass-card rounded-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-white">Available Processing Steps</h2>
            <div id="availableSteps" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                <!-- Steps will be loaded here -->
            </div>
        </div>
        
        <!-- Pipeline Builder Section -->
        <div class="glass-card rounded-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold mb-4 text-white">Pipeline Configuration</h2>
            <div id="pipelineSteps" class="space-y-2 mb-4 min-h-[100px] border-2 border-dashed border-gray-500 rounded-lg p-4">
                <p class="text-gray-400 text-center">No steps added yet. Click on available steps above to add them.</p>
            </div>
            <div class="flex gap-4">
                <button onclick="clearPipeline()" class="btn-danger text-white px-8 py-4 rounded-xl font-semibold flex items-center group">
                    <svg class="w-5 h-5 mr-2 group-hover:rotate-180 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                    </svg>
                    Clear Pipeline
                </button>
                <button onclick="processImage()" class="btn-primary text-white px-8 py-4 rounded-xl font-semibold flex items-center group">
                    <svg class="w-5 h-5 mr-2 group-hover:scale-110 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                    </svg>
                    Process Image
                </button>
            </div>
        </div>
        
        <!-- Results Section -->
        <div id="resultsSection" class="hidden">
            <div class="glass-card rounded-lg p-8">
                <h2 class="text-3xl font-semibold mb-6 text-white">Results</h2>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div>
                        <h3 class="text-xl font-medium mb-3 text-gray-300">Original Image</h3>
                        <div class="image-container border-2 border-gray-600 rounded-lg p-2">
                            <img id="originalImage" src="" alt="Original">
                        </div>
                    </div>
                    <div>
                        <h3 class="text-xl font-medium mb-3 text-gray-300">Processed Image</h3>
                        <div class="image-container border-2 border-gray-600 rounded-lg p-2">
                            <img id="processedImage" src="" alt="Processed">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let availableSteps = {};
        let pipeline = [];
        let uploadedImageBase64 = null;
        
        // Load available steps on page load
        async function loadAvailableSteps() {
            try {
                const response = await fetch('/api/steps');
                availableSteps = await response.json();
                displayAvailableSteps();
            } catch (error) {
                console.error('Error loading steps:', error);
                alert('Failed to load available steps');
            }
        }
        
        // Display available steps
        function displayAvailableSteps() {
            const container = document.getElementById('availableSteps');
            container.innerHTML = '';
            
            // SVG icons for each step type (monochrome)
            const stepIcons = {
                'brightness': '<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd"></path></svg>',
                'saturation': '<svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M4 2a2 2 0 00-2 2v11a3 3 0 106 0V4a2 2 0 00-2-2H4zm1 14a1 1 0 100-2 1 1 0 000 2zm5-1.757l4.9-4.9a2 2 0 000-2.828L13.485 5.1a2 2 0 00-2.828 0L10 5.757v8.486zM16 18H9.071l6-6H16a2 2 0 012 2v2a2 2 0 01-2 2z" clip-rule="evenodd"></path></svg>',
                'hue': '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7h2a2 2 0 012 2v10m0 0a2 2 0 01-2 2h-2m2-2v-5a2 2 0 00-2-2h-2"></path></svg>',
                'box_blur': '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"></path></svg>',
                'unsharp_mask': '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>',
                'crop': '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5"></path></svg>',
                'rotate': '<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>'
            };
            
            for (const [stepName, stepInfo] of Object.entries(availableSteps)) {
                const stepCard = document.createElement('div');
                stepCard.className = 'step-card p-4 cursor-pointer text-white relative group';
                stepCard.onclick = () => addStepToPipeline(stepName);
                
                stepCard.innerHTML = `
                    <div class="param-badge text-xs">${stepInfo.parameters.length}</div>
                    <div class="step-icon w-10 h-10 text-sm mb-3">
                        ${stepIcons[stepName] || '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>'}
                    </div>
                    <h3 class="font-bold text-sm mb-1 bg-gradient-to-r from-gray-100 to-gray-300 bg-clip-text text-transparent">
                        ${stepInfo.name}
                    </h3>
                    <p class="text-xs text-gray-400 leading-tight hidden lg:block">
                        ${stepInfo.description}
                    </p>
                    <div class="mt-2 flex items-center text-xs text-gray-500 group-hover:text-gray-400 transition-colors">
                        <span class="mr-1">Add</span>
                        <svg class="w-3 h-3 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6"></path>
                        </svg>
                    </div>
                `;
                
                container.appendChild(stepCard);
            }
        }
        
        // Add step to pipeline
        function addStepToPipeline(stepName) {
            const stepInfo = availableSteps[stepName];
            const stepConfig = {
                step: stepName,
                params: {}
            };
            
            // Set default parameters
            stepInfo.parameters.forEach(param => {
                stepConfig.params[param.name] = param.default;
            });
            
            pipeline.push(stepConfig);
            displayPipeline();
        }
        
        // Display current pipeline
        function displayPipeline() {
            const container = document.getElementById('pipelineSteps');
            
            if (pipeline.length === 0) {
                container.innerHTML = '<p class="text-gray-400 text-center">No steps added yet. Click on available steps above to add them.</p>';
                return;
            }
            
            container.innerHTML = '';
            
            pipeline.forEach((stepConfig, index) => {
                const stepInfo = availableSteps[stepConfig.step];
                const stepDiv = document.createElement('div');
                stepDiv.className = 'pipeline-item p-4 rounded-lg text-white';
                
                let paramsHtml = '';
                stepInfo.parameters.forEach(param => {
                    const inputId = `param-${index}-${param.name}`;
                    paramsHtml += `
                        <div class="mb-2">
                            <label for="${inputId}" class="block text-sm font-medium text-gray-200">
                                ${param.name} (${param.type})
                            </label>
                            <input
                                type="${param.type === 'int' ? 'number' : 'number'}"
                                id="${inputId}"
                                value="${stepConfig.params[param.name]}"
                                ${param.min_value !== undefined ? `min="${param.min_value}"` : ''}
                                ${param.max_value !== undefined ? `max="${param.max_value}"` : ''}
                                ${param.type === 'float' ? 'step="0.1"' : 'step="1"'}
                                onchange="updateParameter(${index}, '${param.name}', this.value, '${param.type}')"
                                class="mt-1 block w-full rounded-md bg-gray-800 border-gray-600 text-white shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm px-3 py-2"
                            >
                            <span class="text-xs text-gray-400">${param.description}</span>
                        </div>
                    `;
                });
                
                stepDiv.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <h4 class="font-semibold">${index + 1}. ${stepInfo.name}</h4>
                        <button onclick="removeStep(${index})" class="text-red-400 hover:text-red-300">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                    <div class="text-sm text-gray-300 mb-2">${stepInfo.description}</div>
                    ${paramsHtml}
                `;
                
                container.appendChild(stepDiv);
            });
        }
        
        // Update parameter value
        function updateParameter(stepIndex, paramName, value, paramType) {
            if (paramType === 'int') {
                pipeline[stepIndex].params[paramName] = parseInt(value);
            } else if (paramType === 'float') {
                pipeline[stepIndex].params[paramName] = parseFloat(value);
            } else {
                pipeline[stepIndex].params[paramName] = value;
            }
        }
        
        // Remove step from pipeline
        function removeStep(index) {
            pipeline.splice(index, 1);
            displayPipeline();
        }
        
        // Clear entire pipeline
        function clearPipeline() {
            pipeline = [];
            displayPipeline();
        }
        
        // Handle image upload
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    uploadedImageBase64 = e.target.result;
                    // Show preview
                    document.getElementById('resultsSection').classList.remove('hidden');
                    document.getElementById('originalImage').src = uploadedImageBase64;
                    document.getElementById('processedImage').src = '';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Process image with current pipeline
        async function processImage() {
            if (!uploadedImageBase64) {
                alert('Please upload an image first');
                return;
            }
            
            if (pipeline.length === 0) {
                alert('Please add at least one processing step to the pipeline');
                return;
            }
            
            try {
                // Show loading state
                document.getElementById('processedImage').src = '';
                document.getElementById('processedImage').alt = 'Processing...';
                
                const response = await fetch('/api/process-json', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image: uploadedImageBase64,
                        pipeline: pipeline
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                // Display results
                document.getElementById('resultsSection').classList.remove('hidden');
                document.getElementById('originalImage').src = result.original;
                document.getElementById('processedImage').src = result.processed;
                document.getElementById('processedImage').alt = 'Processed';
                
            } catch (error) {
                console.error('Error processing image:', error);
                alert('Failed to process image: ' + error.message);
            }
        }
        
        // Initialize on page load
        loadAvailableSteps();
    </script>
</body>
</html>"""
    return html_content


@app.get("/api/steps")
async def get_available_steps():
    """Return all available image processing steps and their parameters."""
    return pipeline.get_available_steps()


@app.post("/api/process")
async def process_image(file: UploadFile = File(...)):
    """Process an uploaded image with the specified pipeline."""
    try:
        # Read the uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get pipeline configuration from form data
        form = await file.form()
        pipeline_str = form.get("pipeline", "[]")
        pipeline_config = json.loads(pipeline_str)
        
        # Process the image
        processed_image = pipeline.process_image(image, pipeline_config)
        
        # Convert both images to base64
        original_base64 = image_to_base64(image)
        processed_base64 = image_to_base64(processed_image)
        
        return {
            "original": original_base64,
            "processed": processed_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-json")
async def process_image_json(data: Dict[str, Any]):
    """Process an image provided as base64 with the specified pipeline."""
    try:
        # Extract image and pipeline from request
        image_base64 = data.get("image")
        pipeline_config = data.get("pipeline", [])
        
        if not image_base64:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Convert base64 to image
        image = base64_to_image(image_base64)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Process the image
        processed_image = pipeline.process_image(image, pipeline_config)
        
        # Convert both images to base64
        original_base64 = image_to_base64(image)
        processed_base64 = image_to_base64(processed_image)
        
        return {
            "original": original_base64,
            "processed": processed_base64
        }
        
    except Exception as e:
        import traceback
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)