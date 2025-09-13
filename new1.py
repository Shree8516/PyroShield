from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging
from typing import Optional
import time
import os
import requests
import hashlib
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Fire Detection API", version="1.0.0")

# Add CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
MODEL_PATH = "fire_model.h5"
MODEL_LOAD_TIME = None
TARGET_SIZE = (224, 224)
MODEL_HASH = None

# Webhook URL
WEBHOOK_URL = "https://varshh07.app.n8n.cloud/webhook/fire-alert"

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    processing_time: Optional[float] = None
    error: Optional[str] = None
    debug_info: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    load_time: Optional[str] = None
    model_input_shape: Optional[str] = None
    model_output_shape: Optional[str] = None
    model_hash: Optional[str] = None
    file_size: Optional[int] = None

def calculate_file_hash(file_path):
    """Calculate MD5 hash of the model file to detect corruption"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return None

def validate_model_integrity():
    """Perform comprehensive model validation"""
    global model, MODEL_HASH
    
    if model is None:
        return False, "Model not loaded"
    
    try:
        # Test with a simple input to see if model works
        test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Try both preprocessing methods
        test_input_resnet = resnet_preprocess_input(test_input.copy())
        test_input_simple = test_input / 255.0
        
        logger.info(f"Test input shape: {test_input.shape}")
        logger.info(f"ResNet preprocessed range: {test_input_resnet.min():.3f} to {test_input_resnet.max():.3f}")
        logger.info(f"Simple preprocessed range: {test_input_simple.min():.3f} to {test_input_simple.max():.3f}")
        
        # Test prediction with both methods
        try:
            prediction_resnet = model.predict(test_input_resnet, verbose=0)
            logger.info(f"ResNet prediction: {prediction_resnet}")
        except Exception as e:
            logger.error(f"ResNet prediction failed: {e}")
            prediction_resnet = None
            
        try:
            prediction_simple = model.predict(test_input_simple, verbose=0)
            logger.info(f"Simple prediction: {prediction_simple}")
        except Exception as e:
            logger.error(f"Simple prediction failed: {e}")
            prediction_simple = None
        
        # Check if any prediction works
        if prediction_resnet is not None:
            prediction = prediction_resnet
        elif prediction_simple is not None:
            prediction = prediction_simple
        else:
            return False, "Both prediction methods failed"
        
        # Log the actual prediction values for debugging
        logger.info(f"Validation prediction value: {prediction[0][0]}")
        
        # Check prediction validity
        if not isinstance(prediction, np.ndarray):
            return False, "Model prediction is not a numpy array"
        
        if np.any(np.isnan(prediction)):
            return False, "Model produces NaN predictions"
        
        if np.any(np.isinf(prediction)):
            return False, "Model produces infinite predictions"
            
        return True, "Model passed integrity check"
        
    except Exception as e:
        return False, f"Model validation failed: {str(e)}"

def load_model():
    """Load the TensorFlow model with comprehensive validation"""
    global model, MODEL_LOAD_TIME, TARGET_SIZE, MODEL_HASH
    
    try:
        logger.info(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
            
        # Check file size and basic integrity
        file_size = os.path.getsize(MODEL_PATH)
        logger.info(f"Model file size: {file_size} bytes")
        
        if file_size < 1024:  # Less than 1KB is suspicious
            logger.error(f"Model file seems too small: {file_size} bytes")
            return False
        
        # Calculate file hash for corruption detection
        MODEL_HASH = calculate_file_hash(MODEL_PATH)
        logger.info(f"Model file hash: {MODEL_HASH}")
            
        logger.info(f"Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOAD_TIME = time.time()
        
        # Determine the expected input shape from the model
        if model.input_shape[1] is not None and model.input_shape[2] is not None:
            TARGET_SIZE = (model.input_shape[1], model.input_shape[2])
            logger.info(f"Model expects input size: {TARGET_SIZE}")
        else:
            logger.warning("Could not determine model input shape, using default (224, 224)")
            TARGET_SIZE = (224, 224)
        
        # Log model details
        logger.info(f"Model loaded successfully!")
        logger.info(f"Input shape: {model.input_shape}")
        logger.info(f"Output shape: {model.output_shape}")
        logger.info(f"Number of layers: {len(model.layers)}")
        
        # Print last few layers for debugging
        for i, layer in enumerate(model.layers[-3:]):
            logger.info(f"Layer {i}: {layer.name} - {type(layer).__name__}")
            if hasattr(layer, 'activation'):
                logger.info(f"  Activation: {layer.activation}")
            if hasattr(layer, 'units'):
                logger.info(f"  Units: {layer.units}")
        
        # Validate model integrity
        is_valid, validation_msg = validate_model_integrity()
        if not is_valid:
            logger.error(f"Model validation failed: {validation_msg}")
            model = None
            return False
            
        logger.info(f"Model validation passed: {validation_msg}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        return False

@app.on_event("startup")
async def startup_event():
    """Load model when the application starts"""
    if not load_model():
        logger.error("Failed to load model during startup")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fire Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "validate": "/validate-model",
            "test": "/test-prediction"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_valid, validation_msg = validate_model_integrity() if model else (False, "Model not loaded")
    
    model_details = {
        "status": "healthy" if (model and is_valid) else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "load_time": time.ctime(MODEL_LOAD_TIME) if MODEL_LOAD_TIME else None,
        "model_input_shape": str(model.input_shape) if model else None,
        "model_output_shape": str(model.output_shape) if model else None,
        "model_hash": MODEL_HASH,
        "file_size": os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None,
        "validation_message": validation_msg
    }
    return model_details

@app.get("/validate-model")
async def validate_model():
    """Validate model integrity"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    is_valid, message = validate_model_integrity()
    return {
        "valid": is_valid,
        "message": message,
        "model_hash": MODEL_HASH,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape)
    }

@app.get("/test-prediction")
async def test_prediction():
    """Test prediction with a sample image"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create a simple test image (red dominant - should be fire)
        test_image = np.ones((224, 224, 3), dtype=np.uint8)
        test_image[:, :, 0] = 200  # Red channel
        test_image[:, :, 1] = 50   # Green channel
        test_image[:, :, 2] = 50   # Blue channel
        
        logger.info(f"Created test image shape: {test_image.shape}")
        logger.info(f"Test image range: {test_image.min()} to {test_image.max()}")
        
        # Convert to bytes and process
        img_pil = Image.fromarray(test_image)
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        logger.info(f"Image bytes length: {len(img_byte_arr)}")
        
        # Try both preprocessing methods
        results = {}
        
        # Method 1: ResNet preprocessing
        try:
            x_resnet = preprocess_image(img_byte_arr, TARGET_SIZE)
            logger.info(f"ResNet preprocessed shape: {x_resnet.shape}")
            logger.info(f"ResNet preprocessed range: {x_resnet.min():.3f} to {x_resnet.max():.3f}")
            
            preds_resnet = model.predict(x_resnet, verbose=0)
            logger.info(f"ResNet raw predictions: {preds_resnet}")
            
            # Process predictions - Binary classification
            confidence = float(preds_resnet[0][0]) * 100
            label = "fire" if preds_resnet[0][0] > 0.5 else "no_fire"
            
            results["resnet_preprocessing"] = {
                "label": label,
                "confidence": round(confidence, 2),
                "raw_predictions": preds_resnet.tolist(),
                "shape": str(preds_resnet.shape)
            }
            
        except Exception as e:
            results["resnet_preprocessing"] = {"error": str(e)}
            logger.error(f"ResNet preprocessing failed: {e}")
        
        # Method 2: Simple normalization
        try:
            img = Image.open(io.BytesIO(img_byte_arr)).convert("RGB")
            img = img.resize(TARGET_SIZE)
            img_array = np.array(img)
            img_array_expanded = np.expand_dims(img_array, axis=0)
            x_simple = img_array_expanded / 255.0
            
            logger.info(f"Simple preprocessed shape: {x_simple.shape}")
            logger.info(f"Simple preprocessed range: {x_simple.min():.3f} to {x_simple.max():.3f}")
            
            preds_simple = model.predict(x_simple, verbose=0)
            logger.info(f"Simple raw predictions: {preds_simple}")
            
            # Process predictions - Binary classification
            confidence = float(preds_simple[0][0]) * 100
            label = "fire" if preds_simple[0][0] > 0.5 else "no_fire"
            
            results["simple_preprocessing"] = {
                "label": label,
                "confidence": round(confidence, 2),
                "raw_predictions": preds_simple.tolist(),
                "shape": str(preds_simple.shape)
            }
            
        except Exception as e:
            results["simple_preprocessing"] = {"error": str(e)}
            logger.error(f"Simple preprocessing failed: {e}")
        
        return {
            "test_image": "red_dominant_should_be_fire",
            "results": results,
            "model_input_shape": str(model.input_shape),
            "model_output_shape": str(model.output_shape)
        }
        
    except Exception as e:
        logger.error(f"Test prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test prediction failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get detailed information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    layer_info = []
    for i, layer in enumerate(model.layers):
        layer_info.append({
            "index": i,
            "name": layer.name,
            "type": type(layer).__name__,
            "output_shape": str(layer.output_shape),
            "trainable": layer.trainable
        })
    
    return {
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "layers": len(model.layers),
        "target_size": TARGET_SIZE,
        "layer_details": layer_info[-5:],  # Last 5 layers
        "last_layer_activation": str(getattr(model.layers[-1], 'activation', 'None'))
    }

@app.post("/reload-model")
async def reload_model():
    """Reload the model"""
    success = load_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

def preprocess_image(image_bytes, target_size=TARGET_SIZE):
    """Preprocess image for model prediction."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # Use simple normalization instead of ResNet preprocessing
        preprocessed_image = img_array_expanded / 255.0  # ← CHANGED THIS LINE
        
        return preprocessed_image
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def trigger_webhook(label: str, confidence: float, processing_time: float):
    """Trigger webhook if fire is detected"""
    if label == "fire":
        logger.info("🔥 Fire detected. Triggering webhook...")
        try:
            # Prepare the data payload for the webhook - with null coordinates
            payload = {
                "label": label,
                "confidence": round(confidence, 2),
                "processing_time_ms": round(processing_time * 1000, 2),
                "timestamp": time.time(),
                "latitude": None,  # ← Set to null/empty
                "longitude": None, # ← Set to null/empty
                "message": "Fire detected with high confidence."
            }
            
            response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"✅ Webhook triggered successfully. Status code: {response.status_code}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to trigger webhook: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while triggering the webhook: {e}")
            return False
    return False

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict if an image contains fire"""
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Save debug image
        debug_filename = f"debug_{int(time.time())}.jpg"
        with open(debug_filename, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Saved debug image: {debug_filename}")
        
        x = preprocess_image(image_bytes, TARGET_SIZE)
        
        preds = model.predict(x, verbose=0)
        processing_time = time.time() - start_time

        # Debug information
        debug_info = {
            "raw_predictions": preds.tolist(),
            "prediction_shape": str(preds.shape),
            "preprocessing_time": processing_time
        }

        # FIXED: Binary classification with sigmoid activation
        # Output is single probability value [0, 1] where:
        # > 0.5 = fire, < 0.5 = no_fire
        confidence = float(preds[0][0]) * 100  # Convert to percentage
        label = "fire" if preds[0][0] > 0.5 else "no_fire"

        logger.info(f"Raw prediction value: {preds[0][0]}")
        logger.info(f"Prediction: {label} ({confidence:.2f}%) in {processing_time:.3f}s")
        
        # Trigger the webhook if fire is detected
        trigger_webhook(label, confidence, processing_time)

        return PredictionResponse(
            label=label, 
            confidence=round(confidence, 2),
            processing_time=round(processing_time, 3),
            debug_info=debug_info
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")