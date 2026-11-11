from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
from werkzeug.utils import secure_filename
from gradio_client import Client, handle_file

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'},
    'audio': {'wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a'}
}

# Create upload folder
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Hugging Face Space client
try:
    hf_client = Client("PraneshJs/fakevideodetect")
    HF_AVAILABLE = True
    print("✅ Hugging Face Space client initialized successfully")
except Exception as e:
    HF_AVAILABLE = False
    print(f"⚠️ Hugging Face Space client failed to initialize: {e}")

def allowed_file(filename, file_type):
    """Check if file extension is allowed for the given type"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS.get(file_type, set())

def get_file_info(file):
    """Get file information"""
    try:
        file.seek(0, 2)  # Seek to end
        size = file.tell()
        file.seek(0)     # Reset to beginning
        return {
            'name': file.filename,
            'size': size,
            'size_mb': round(size / (1024 * 1024), 2),
            'size_kb': round(size / 1024, 1)
        }
    except:
        return {'name': file.filename, 'size': 0, 'size_mb': 0, 'size_kb': 0}

def predict_with_hf_space(file_path, prediction_type):
    """Make prediction using Hugging Face Space"""
    try:
        if not HF_AVAILABLE:
            raise Exception("Hugging Face Space client not available")
        
        if prediction_type == 'video':
            result = hf_client.predict(
                input_video={"video": handle_file(file_path)},
                api_name="/predict_1"
            )
        elif prediction_type == 'audio':
            result = hf_client.predict(
                input_audio=handle_file(file_path),
                api_name="/predict_2"
            )
        else:
            raise Exception(f"Unsupported prediction type: {prediction_type}")
        
        return result
    except Exception as e:
        print(f"❌ HF Space prediction error: {e}")
        raise e

@app.route('/api/predict/<prediction_type>', methods=['POST'])
def predict(prediction_type):
    """Prediction endpoint for video and audio"""
    try:
        # Validate prediction type
        if prediction_type not in ['video', 'audio']:
            return jsonify({'error': 'Invalid prediction type. Use: video or audio'}), 400

        if not HF_AVAILABLE:
            return jsonify({'error': 'Prediction service is unavailable.'}), 503
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not allowed_file(file.filename, prediction_type):
            allowed_exts = ', '.join(ALLOWED_EXTENSIONS[prediction_type])
            return jsonify({'error': f'Invalid file type for {prediction_type}. Allowed: {allowed_exts}'}), 400
        
        # Get file info
        file_info = get_file_info(file)
        print(f"📁 Processing {prediction_type}: {file_info['name']} ({file_info['size_mb']}MB)")
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time() * 1000))
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        print(f"💾 File saved to: {file_path}")
        
        try:
            print("Using Hugging Face Space for prediction...")
            result = predict_with_hf_space(file_path, prediction_type)
            return jsonify({
                'result': result,
                'status': 'success',
                'source': 'huggingface_space',
                'file_info': file_info
            })
                
        except Exception as prediction_error:
            print(f"❌ Prediction error: {prediction_error}")
            return jsonify({
                'error': 'Model prediction failed.'
            }), 502
        
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ Cleaned up: {file_path}")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup error: {cleanup_error}")
    
    except Exception as e:
        print(f"❌ Server error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/examples/<prediction_type>', methods=['GET'])
def get_examples(prediction_type):
    """Get example files for testing"""
    examples = {
        'video': [
            {'name': 'REAL.mp4', 'path': 'videos/REAL.mp4', 'description': 'Real video sample'},
            {'name': 'FAKE.mp4', 'path': 'videos/FAKE.mp4', 'description': 'Synthetic video sample'}
        ],
        'audio': [
            {'name': 'REAL.mp3', 'path': 'audios/REAL.mp3', 'description': 'Audio sample'},
            {'name': 'Synthetic speech.mp3', 'path': 'audios/ElevenLabs_2026-11-05T03_44_44_Roger - Laid-Back, Casual, Resonant_pre_s35_sim65_v4.mp3', 'description': 'Synthetic audio sample'}
        ]
    }

    if prediction_type not in examples:
        return jsonify({'error': 'Invalid prediction type. Use: video or audio'}), 400
    
    return jsonify({
        'examples': examples[prediction_type],
        'status': 'success'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Add OPTIONS handler for CORS preflight
@app.route('/api/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    return '', 200

if __name__ == '__main__':
    print("🚀 Starting AI Video Detector Backend...")
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📊 Max file size: {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB")
    print(f"🤖 Hugging Face Space: {'Available' if HF_AVAILABLE else 'Not Available'}")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"🌐 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)