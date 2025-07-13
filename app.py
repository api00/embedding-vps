from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import logging
import os
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None

def load_model():
    """Load the Nomic embedding model"""
    global model
    try:
        logger.info("Loading Nomic embedding model...")
        # Use the Nomic embed text model
        model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', 
                                   cache_folder='/app/models',
                                   trust_remote_code=True)
        logger.info("Nomic model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy', 'model': 'nomic-embed-text-v1'}), 200

@app.route('/embed', methods=['POST'])
def embed_text():
    """Generate embeddings for text"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get text from request
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Handle empty text
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Generate embedding with Nomic model
        start_time = time.time()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        processing_time = time.time() - start_time
        
        logger.info(f"Generated embedding for text length: {len(text)}, time: {processing_time:.3f}s")
        
        return jsonify({
            'embedding': embedding.tolist(),
            'dimensions': len(embedding),
            'processing_time': processing_time,
            'model': 'nomic-embed-text-v1'
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch():
    """Generate embeddings for multiple texts"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get texts from request
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be a list'}), 400
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            return jsonify({'error': 'No valid texts provided'}), 400
        
        # Generate embeddings with batch processing
        start_time = time.time()
        embeddings = model.encode(valid_texts, 
                                 convert_to_numpy=True, 
                                 normalize_embeddings=True,
                                 batch_size=32)
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(embeddings)} embeddings, time: {processing_time:.3f}s")
        
        return jsonify({
            'embeddings': embeddings.tolist(),
            'count': len(embeddings),
            'dimensions': len(embeddings[0]) if len(embeddings) > 0 else 0,
            'processing_time': processing_time,
            'model': 'nomic-embed-text-v1'
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Nomic Embedding API',
        'model': 'nomic-embed-text-v1',
        'dimensions': 768,
        'endpoints': {
            'health': 'GET /health',
            'single_embedding': 'POST /embed',
            'batch_embedding': 'POST /embed/batch'
        },
        'status': 'ready' if model else 'loading'
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
