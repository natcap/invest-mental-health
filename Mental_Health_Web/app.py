from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
import json
import tempfile
import shutil
from werkzeug.utils import secure_filename
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import base64
import io
import numpy as np
import pandas as pd
from Treecover_approach import run_ndvi_tree_analysis, run_pd_analysis, plot_ndvi_vs_negoal_gradient
import zipfile
import geopandas as gpd

# Add ngrok support
try:
    from pyngrok import ngrok

    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("Ngrok not installed. Run: pip install pyngrok")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

ALLOWED_EXTENSIONS = {'tif', 'tiff', 'shp', 'shx', 'dbf', 'prj', 'cpg', 'xlsx', 'xls', 'geojson'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def figure_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url


def convert_shapefile_to_geojson(shp_path):
    """Convert shapefile to GeoJSON with multiple fallback strategies"""
    try:
        print(f"Converting shapefile to GeoJSON: {shp_path}")

        # Strategy 1: Direct read (most common case)
        try:
            gdf = gpd.read_file(shp_path)
            print(f"Strategy 1 success - Shape: {gdf.shape}, Columns: {list(gdf.columns)}")
        except Exception as e1:
            print(f"Strategy 1 failed: {e1}")

            # Strategy 2: Set GDAL environment variables
            try:
                os.environ['SHAPE_RESTORE_SHX'] = 'YES'
                os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
                gdf = gpd.read_file(shp_path)
                print(f"Strategy 2 success with GDAL options - Shape: {gdf.shape}")
            except Exception as e2:
                print(f"Strategy 2 failed: {e2}")

                # Strategy 3: Copy to simple path and retry
                try:
                    temp_dir = os.path.dirname(shp_path)
                    simple_shp = os.path.join(temp_dir, "temp_simple.shp")

                    # Copy all shapefile components with simple names
                    base_orig = shp_path.replace('.shp', '')
                    base_simple = simple_shp.replace('.shp', '')

                    copied_files = []
                    for ext in ['shp', 'shx', 'dbf', 'prj', 'cpg']:
                        orig_file = f"{base_orig}.{ext}"
                        simple_file = f"{base_simple}.{ext}"
                        if os.path.exists(orig_file):
                            shutil.copy2(orig_file, simple_file)
                            copied_files.append(simple_file)

                    gdf = gpd.read_file(simple_shp)
                    print(f"Strategy 3 success with simple path - Shape: {gdf.shape}")

                    # Clean up temporary files
                    for temp_file in copied_files:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

                except Exception as e3:
                    print(f"Strategy 3 failed: {e3}")
                    return None

        # Verify GEOID column
        if 'GEOID' in gdf.columns:
            print(f"GEOID found. Sample values: {gdf['GEOID'].head().tolist()}")
        else:
            print(f"Warning: No GEOID column. Available columns: {list(gdf.columns)}")

        # Convert to GeoJSON
        geojson_path = shp_path.replace('.shp', '.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')

        # Verify conversion
        gdf_verify = gpd.read_file(geojson_path)
        print(f"GeoJSON created - Shape: {gdf_verify.shape}, Columns: {list(gdf_verify.columns)}")

        if 'GEOID' in gdf.columns and 'GEOID' not in gdf_verify.columns:
            print("ERROR: GEOID lost during conversion!")
            return None

        print(f"✅ Conversion successful: {geojson_path}")
        return geojson_path

    except Exception as e:
        print(f"All conversion strategies failed: {e}")
        return None


class WebAnalysisSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        self.results_dir = os.path.join('static/results', session_id)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Analysis data
        self.x_lowess = None
        self.y_lowess = None
        self.aoi_adm2 = None
        self.ndvi_resampled_path = None
        self.ne_goal = 0.3

    def get_file_path(self, file_type):
        """Get the path for uploaded file of specific type"""
        file_mapping = {
            'aoi_county': 'aoi_county.shp',
            'aoi_tract': 'aoi_tract.shp',
            'population': 'population.tif',
            'ndvi': 'ndvi.tif',
            'tree_cover': 'tree_cover.tif',
            'risk': 'risk.tif',
            'health_effects': 'health_effects.xlsx'
        }
        return os.path.join(self.upload_dir, file_mapping.get(file_type, ''))

    def find_spatial_file(self, file_type):
        """Find any spatial file (.shp or .geojson) for the given type"""
        try:
            files = os.listdir(self.upload_dir)

            # First look for GeoJSON files (more reliable)
            geojson_files = [f for f in files if f.endswith('.geojson')]
            for geojson_file in geojson_files:
                geojson_path = os.path.join(self.upload_dir, geojson_file)
                try:
                    # Verify it's readable
                    test_gdf = gpd.read_file(geojson_path)
                    print(f"Found valid GeoJSON: {geojson_path}")
                    return geojson_path
                except Exception as e:
                    print(f"Invalid GeoJSON {geojson_path}: {e}")
                    continue

            # If no GeoJSON, look for shapefiles
            shp_files = [f for f in files if f.endswith('.shp')]
            for shp_file in shp_files:
                shp_path = os.path.join(self.upload_dir, shp_file)
                try:
                    # Try to convert to GeoJSON
                    geojson_path = convert_shapefile_to_geojson(shp_path)
                    if geojson_path:
                        return geojson_path
                except Exception as e:
                    print(f"Failed to process shapefile {shp_path}: {e}")
                    continue

        except Exception as e:
            print(f"Error finding spatial file: {e}")

        return None


# Store active sessions
sessions = {}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/create_session', methods=['POST'])
def create_session():
    """Create a new analysis session"""
    import uuid
    session_id = str(uuid.uuid4())
    sessions[session_id] = WebAnalysisSession(session_id)
    return jsonify({'session_id': session_id, 'status': 'success'})


@app.route('/upload_file/<session_id>/<file_type>', methods=['POST'])
def upload_file(session_id, file_type):
    """Upload a file for the analysis"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    files = request.files.getlist('file')
    uploaded_files = []

    for file in files:
        if file.filename == '':
            continue

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(session.upload_dir, filename)
            file.save(file_path)
            uploaded_files.append(filename)

            # Special handling for spatial files (shapefile or geojson)
            if file_type in ['aoi_county', 'aoi_tract'] and filename.endswith('.shp'):
                print(f"Processing shapefile: {filename}")

                # Convert to GeoJSON for reliability
                geojson_path = convert_shapefile_to_geojson(file_path)
                if geojson_path:
                    print(f"Shapefile converted successfully to GeoJSON")
                    # Create standard named copy
                    standard_path = session.get_file_path(file_type).replace('.shp', '.geojson')
                    try:
                        shutil.copy2(geojson_path, standard_path)
                    except Exception as e:
                        print(f"Error copying GeoJSON: {e}")
                else:
                    return jsonify({'error': f'Failed to process shapefile: {filename}'}), 400

            elif file_type in ['aoi_county', 'aoi_tract'] and filename.endswith('.geojson'):
                print(f"Processing GeoJSON: {filename}")
                # Verify GeoJSON
                try:
                    test_gdf = gpd.read_file(file_path)
                    print(f"GeoJSON verified - Columns: {list(test_gdf.columns)}")
                    if 'GEOID' not in test_gdf.columns:
                        print("Warning: No GEOID column found in GeoJSON")

                    # Create standard named copy
                    standard_path = session.get_file_path(file_type).replace('.shp', '.geojson')
                    shutil.copy2(file_path, standard_path)
                except Exception as e:
                    return jsonify({'error': f'Invalid GeoJSON file: {filename}'}), 400

            # For other file types, create standard named copy
            elif file_type not in ['aoi_county', 'aoi_tract']:
                standard_path = session.get_file_path(file_type)
                try:
                    shutil.copy2(file_path, standard_path)
                    print(f"Copied {filename} to {standard_path}")
                except Exception as e:
                    print(f"Error copying file {filename}: {e}")
                    # If copy fails, keep track of original file
                    pass

    return jsonify({
        'status': 'success',
        'uploaded_files': uploaded_files,
        'message': f'Uploaded {len(uploaded_files)} files for {file_type}'
    })


@app.route('/run_analysis/<session_id>/step1', methods=['POST'])
def run_analysis_step1(session_id):
    """Run the first step of analysis (NDVI + Tree Cover + LOWESS)"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]

    try:
        # Prepare file paths with improved checking
        file_types = ['aoi_county', 'aoi_tract', 'population', 'ndvi', 'tree_cover', 'risk', 'health_effects']
        paths = []
        missing_files = []

        for file_type in file_types:
            if file_type in ['aoi_county', 'aoi_tract']:
                # Look for spatial files (GeoJSON or shapefile)
                spatial_file = session.find_spatial_file(file_type)
                if spatial_file and os.path.exists(spatial_file):
                    # Test read the file
                    try:
                        test_gdf = gpd.read_file(spatial_file)
                        paths.append(spatial_file)
                        print(f"Using spatial file: {spatial_file}")
                    except Exception as e:
                        print(f"Spatial file read test failed: {e}")
                        missing_files.append(f"{file_type} (corrupted)")
                        paths.append('')
                else:
                    missing_files.append(file_type)
                    paths.append('')
            else:
                # For non-spatial files, check both standard path and uploaded files
                standard_path = session.get_file_path(file_type)
                found_file = None

                # First check if standard named file exists
                if os.path.exists(standard_path):
                    found_file = standard_path
                else:
                    # Check for any uploaded file that matches the type
                    try:
                        files = os.listdir(session.upload_dir)
                        for filename in files:
                            file_lower = filename.lower()
                            if file_type == 'tree_cover' and (
                                    'tree' in file_lower and filename.endswith(('.tif', '.tiff'))):
                                found_file = os.path.join(session.upload_dir, filename)
                                break
                            elif file_type == 'population' and (
                                    'pop' in file_lower and filename.endswith(('.tif', '.tiff'))):
                                found_file = os.path.join(session.upload_dir, filename)
                                break
                            elif file_type == 'ndvi' and (
                                    'ndvi' in file_lower and filename.endswith(('.tif', '.tiff'))):
                                found_file = os.path.join(session.upload_dir, filename)
                                break
                            elif file_type == 'risk' and (
                                    ('risk' in file_lower or 'depress' in file_lower) and filename.endswith(
                                    ('.tif', '.tiff'))):
                                found_file = os.path.join(session.upload_dir, filename)
                                break
                            elif file_type == 'health_effects' and filename.endswith(('.xlsx', '.xls')):
                                found_file = os.path.join(session.upload_dir, filename)
                                break
                    except Exception as e:
                        print(f"Error searching for {file_type}: {e}")

                if found_file:
                    paths.append(found_file)
                    print(f"Using {file_type} file: {found_file}")
                else:
                    missing_files.append(file_type)
                    paths.append('')
                    print(f"Missing {file_type} file, checked: {standard_path}")

        # Add output directory
        paths.append(session.results_dir)

        if missing_files:
            return jsonify({'error': f'Missing or corrupted files: {", ".join(missing_files)}'}), 400

        print(f"Running analysis with paths: {paths}")

        # Run analysis
        ne_goal, ndvi_fig, tree_fig, slider_fig, x_lowess, y_lowess, aoi_adm2, ndvi_resampled_path = run_ndvi_tree_analysis(
            *paths)

        # Calculate current average tree cover from the analysis data
        try:
            # Read the tree cover raster that was processed in the analysis
            tree_cover_path = None
            for path in paths:
                if 'tree' in path.lower() and path.endswith('.tif'):
                    tree_cover_path = path
                    break

            current_tree_cover = 0.0
            if tree_cover_path and os.path.exists(tree_cover_path):
                import rasterio
                with rasterio.open(tree_cover_path) as src:
                    data = src.read(1, masked=True)
                    # Calculate mean excluding nodata values
                    if data.compressed().size > 0:
                        current_tree_cover = float(data.compressed().mean())
                        # If values are 0-255 scale, convert to percentage
                        if current_tree_cover > 100:
                            current_tree_cover = current_tree_cover * 100.0 / 255.0
                        current_tree_cover = max(0.0, min(100.0, current_tree_cover))

            print(f"Calculated current tree cover: {current_tree_cover:.2f}%")

        except Exception as e:
            print(f"Error calculating current tree cover: {e}")
            current_tree_cover = 0.0

        # Store results in session
        session.x_lowess = x_lowess.tolist()
        session.y_lowess = y_lowess.tolist()
        session.aoi_adm2 = aoi_adm2
        session.ndvi_resampled_path = ndvi_resampled_path
        session.ne_goal = ne_goal

        # Convert figures to base64
        results = {
            'status': 'success',
            'ne_goal': float(ne_goal),
            'current_tree_cover': round(current_tree_cover, 2),  # Add current tree cover
            'ndvi_plot': figure_to_base64(ndvi_fig),
            'tree_plot': figure_to_base64(tree_fig),
            'slider_plot': figure_to_base64(slider_fig),
            'x_lowess': session.x_lowess,
            'y_lowess': session.y_lowess,
            'x_min': float(min(x_lowess)),
            'x_max': float(max(x_lowess))
        }

        return jsonify(results)

    except Exception as e:
        import traceback
        print(f"Analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/update_ne_goal/<session_id>', methods=['POST'])
def update_ne_goal(session_id):
    """Update NE_goal based on tree cover selection"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]

    try:
        tree_cover = float(request.json.get('tree_cover', 30))

        if session.x_lowess is None or session.y_lowess is None:
            return jsonify({'error': 'Analysis step 1 not completed'}), 400

        # Calculate NDVI from tree cover using interpolation
        ndvi_val = float(np.interp(tree_cover, session.x_lowess, session.y_lowess))
        ndvi_val = max(0.0, min(1.0, ndvi_val))  # Clamp to valid NDVI range

        session.ne_goal = ndvi_val

        # Check if extrapolated
        min_data_cover = min(session.x_lowess)
        max_data_cover = max(session.x_lowess)
        is_extrapolated = tree_cover < min_data_cover or tree_cover > max_data_cover

        return jsonify({
            'status': 'success',
            'tree_cover': tree_cover,
            'ndvi_value': ndvi_val,
            'is_extrapolated': is_extrapolated
        })

    except Exception as e:
        return jsonify({'error': f'Update failed: {str(e)}'}), 500


@app.route('/run_analysis/<session_id>/step2', methods=['POST'])
def run_analysis_step2(session_id):
    """Run the second step of analysis (PD Analysis)"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]

    try:
        # Get cost value from request
        cost_value = float(request.json.get('cost_value', 11000))

        if session.x_lowess is None or session.aoi_adm2 is None:
            return jsonify({'error': 'Analysis step 1 not completed'}), 400

        # Prepare file paths (same as step1)
        file_types = ['aoi_county', 'aoi_tract', 'population', 'ndvi', 'tree_cover', 'risk', 'health_effects']
        paths = []

        for file_type in file_types:
            if file_type in ['aoi_county', 'aoi_tract']:
                spatial_file = session.find_spatial_file(file_type)
                if spatial_file and os.path.exists(spatial_file):
                    paths.append(spatial_file)
                else:
                    paths.append(session.get_file_path(file_type))
            else:
                paths.append(session.get_file_path(file_type))

        paths.append(session.results_dir)

        # Run PD analysis
        fig1, fig2, fig_hist, fig_cost_curve, total_cases = run_pd_analysis(
            *paths, session.ne_goal, session.aoi_adm2,
            np.array(session.x_lowess), np.array(session.y_lowess), cost_value
        )

        # Generate NDVI gradient plot
        fig3 = plot_ndvi_vs_negoal_gradient(session.ndvi_resampled_path, session.aoi_adm2, session.ne_goal)

        # Calculate summary
        tree_cover_percent = float(np.interp(session.ne_goal, session.y_lowess, session.x_lowess))

        # Fix: Don't multiply by cost_value again if run_pd_analysis already calculated costs
        # total_cases should be the number of preventable cases, not cost amount
        # But check if it's already been multiplied in Treecover_approach.py

        # Simple fix: assume total_cases is case count, not cost
        money_saved = total_cases * cost_value

        # Debug to see what we're getting
        print(f"total_cases from analysis: {total_cases}")
        print(f"cost_value: {cost_value}")
        print(f"calculated money_saved: {money_saved}")

        results = {
            'status': 'success',
            'pd_map_v1': figure_to_base64(fig1),
            'pd_map_v3': figure_to_base64(fig2),
            'cost_histogram': figure_to_base64(fig_hist),
            'cost_curve': figure_to_base64(fig_cost_curve),
            'ndvi_gradient': figure_to_base64(fig3),
            'summary': {
                'tree_cover_percent': round(tree_cover_percent, 1),
                'total_cases': int(total_cases),
                'money_saved': int(total_cases),  # Fix: Use total_cases directly if it's already cost-adjusted
                'cost_value': cost_value
            }
        }

        return jsonify(results)

    except Exception as e:
        import traceback
        print(f"Step 2 analysis error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/download_results/<session_id>')
def download_results(session_id):
    """Download all results as a ZIP file"""
    if session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    session = sessions[session_id]

    # Create a temporary ZIP file
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        # Add all files from results directory
        for root, dirs, files in os.walk(session.results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, session.results_dir)
                zipf.write(file_path, arcname)

    return send_file(temp_zip.name, as_attachment=True, download_name=f'analysis_results_{session_id}.zip')


if __name__ == '__main__':
    port = 5000

    if NGROK_AVAILABLE:
        # Configure authtoken
        ngrok.set_auth_token("31nLXYWVkBbChgDvViA0yXyMZlq_7DK1wjxomEQfPUNR1Nu8K")
        # Start ngrok tunnel
        public_tunnel = ngrok.connect(port)
        print(f"")
        print(f"Public URL: {public_tunnel.public_url}")
        print(f"Share this URL with anyone to access your app!")
        print(f"Local URL: http://localhost:{port}")
        print(f"")

        # Run for external access
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print(f"Ngrok not available. Install with: pip install pyngrok")
        print(f"Local URL: http://localhost:{port}")
        app.run(debug=True, host='0.0.0.0', port=port)