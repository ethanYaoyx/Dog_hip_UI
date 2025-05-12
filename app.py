from flask import Flask, request, render_template, send_file, jsonify, redirect, session
import sqlite3

import os
import scipy.io as sio
import numpy as np
from werkzeug.utils import secure_filename
from model import predict_keypoints
import io
import zipfile
import shutil


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.secret_key = 'your_super_secret_key'  #  用于 session 加密


# === Global constants ===

STATIC_ROOT = os.path.join(os.path.dirname(__file__), 'static')
DB_PATH = os.path.join(STATIC_ROOT, 'user_auth.db')
UPLOAD_FOLDER = os.path.join(STATIC_ROOT, 'uploads')
PRED_FOLDER = os.path.join(STATIC_ROOT, 'Pred')
HUMAN_LABEL_FOLDER = os.path.join(STATIC_ROOT, 'Human_Labels')

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PRED_FOLDER, exist_ok=True)
os.makedirs(HUMAN_LABEL_FOLDER, exist_ok=True)

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect('/login_page')
    return render_template('index.html')


@app.route('/')
def root():
    return redirect('/login_page') 

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login_page')

def get_user_folder(base_folder):
    username = session.get('username', 'default')
    user_folder = os.path.join(base_folder, username)
    os.makedirs(user_folder, exist_ok=True)
    return user_folder


# @app.route('/predict_all', methods=['POST'])
# def predict_all():
#     #print("11111111111111111111111111111")
#     files = request.files.getlist('image')
#     if not files:
#         return "No images uploaded", 400

#     original_path = request.form.get('original_path')
#     if not original_path:
#         return "No original path provided", 400

#     for file in files:
#         #print("222222222222222222")
#         #filename = secure_filename(file.filename)
#         filename = file.filename
#         temp_path = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(temp_path)

#         name, _ = os.path.splitext(filename)
#         pred_path = os.path.join(PRED_FOLDER, f"{name}.mat")
#         print("pred path:", temp_path)
#         if os.path.exists(pred_path):
#             continue

#         try:
#             predict_keypoints(temp_path, result_image_dir=None, label_save_dir=PRED_FOLDER)
#         except Exception as e:
#             print(f"[ERROR] Failed to predict {filename}: {e}")
#             continue

#     return jsonify({"status": "done"})

@app.route('/predict_all', methods=['POST'])
def predict_all():
    
    if 'username' not in session:
        return "Unauthorized", 401

    username = session['username']
    files = request.files.getlist('image')
    if not files:
        return "No images uploaded", 400

    original_path = request.form.get('original_path')
    if not original_path:
        return "No original path provided", 400

    # 创建用户子文件夹
    user_upload_dir = os.path.join(UPLOAD_FOLDER, username)
    user_pred_dir = os.path.join(PRED_FOLDER, username)
    os.makedirs(user_upload_dir, exist_ok=True)
    os.makedirs(user_pred_dir, exist_ok=True)

    for file in files:
        filename = file.filename
        temp_path = os.path.join(user_upload_dir, filename)
        file.save(temp_path)

        name, _ = os.path.splitext(filename)
        pred_path = os.path.join(user_pred_dir, f"{name}.mat")
        
        print("pred path:", pred_path)
        if os.path.exists(pred_path):
            continue

        try:
            predict_keypoints(temp_path, result_image_dir=None, label_save_dir=user_pred_dir)
        except Exception as e:
            print(f"[ERROR] Failed to predict {filename}: {e}")
            continue

    return jsonify({"status": "done"})


# @app.route('/save_label', methods=['POST'])
# def save_label():
#     data = request.get_json()
#     fileonly = data.get('fileonly')
#     points = data.get('points', [])
#     radii = data.get('radii', [])
#     angles = data.get('angles', [])

#     save_path = os.path.join(HUMAN_LABEL_FOLDER, os.path.splitext(fileonly)[0] + '.mat')
#     mat_data = {
#         "Four_points": np.array(points + [radii]),
#         "Angles": np.array([angles])
#     }
#     sio.savemat(save_path, mat_data)
#     print("save label:",save_path)
#     return jsonify({"path": save_path})

@app.route('/save_label', methods=['POST'])
def save_label():
    if 'username' not in session:
        return "Unauthorized", 401

    username = session['username']
    data = request.get_json()
    fileonly = data.get('fileonly')
    points = data.get('points', [])
    radii = data.get('radii', [])
    angles = data.get('angles', [])

    # 构造保存路径
    user_label_dir = os.path.join(HUMAN_LABEL_FOLDER, username)
    os.makedirs(user_label_dir, exist_ok=True)

    save_path = os.path.join(user_label_dir, os.path.splitext(fileonly)[0] + '.mat')
    mat_data = {
        "Four_points": np.array(points + [radii]),
        "Angles": np.array([angles])
    }

    sio.savemat(save_path, mat_data)
    print("save label:", save_path)
    return jsonify({"path": save_path})


# @app.route('/load_label', methods=['POST'])
# def load_label():
#     data = request.get_json()
#     folder_name = data.get('filename')  # 'Pred' or 'Human_Labels'
#     fileonly = data.get('fileonly')

#     folder_map = {
#         'Pred': PRED_FOLDER,
#         'Human_Labels': HUMAN_LABEL_FOLDER
#     }
#     mat_dir = folder_map.get(folder_name, '')
#     mat_path = os.path.join(mat_dir, os.path.splitext(fileonly)[0] + '.mat')
#     print("Load label:",mat_path)
#     if not os.path.exists(mat_path):
#         print('not found')
#         return jsonify({'found': False})

#     mat = sio.loadmat(mat_path)
#     points = mat.get('Four_points', np.zeros((5, 2))).tolist()
#     angles = mat.get('Angles', [[0, 0]])[0]
#     print("angles",angles)

#     return jsonify({
#         'found': True,
#         'points': points[:4],
#         'radii': points[4],
#         'angles': angles.tolist()
#     })
@app.route('/load_label', methods=['POST'])
def load_label():
    if 'username' not in session:
        return "Unauthorized", 401

    username = session['username']
    data = request.get_json()
    folder_name = data.get('filename')  # 'Pred' or 'Human_Labels'
    fileonly = data.get('fileonly')

    folder_map = {
        'Pred': os.path.join(PRED_FOLDER, username),
        'Human_Labels': os.path.join(HUMAN_LABEL_FOLDER, username)
    }
    mat_dir = folder_map.get(folder_name, '')
    mat_path = os.path.join(mat_dir, os.path.splitext(fileonly)[0] + '.mat')
    print("Load label:", mat_path)

    if not os.path.exists(mat_path):
        print('not found')
        return jsonify({'found': False})

    mat = sio.loadmat(mat_path)
    points = mat.get('Four_points', np.zeros((5, 2))).tolist()
    angles = mat.get('Angles', [[0, 0]])[0]
    print("angles", angles)

    return jsonify({
        'found': True,
        'points': points[:4],
        'radii': points[4],
        'angles': angles.tolist()
    })

# @app.route('/download_predicted', methods=['GET'])
# def download_predicted():
#     pred_folder = PRED_FOLDER  
#     if not os.path.exists(pred_folder):
#         return "Predicted folder not found", 404

#     zip_stream = io.BytesIO()
#     with zipfile.ZipFile(zip_stream, 'w') as zipf:
#         for filename in os.listdir(pred_folder):
#             file_path = os.path.join(pred_folder, filename)
#             if os.path.isfile(file_path):
#                 zipf.write(file_path, arcname=filename)

#     zip_stream.seek(0)
#     return send_file(zip_stream, mimetype='application/zip', as_attachment=True, download_name='Predicted_Labels.zip')

@app.route('/download_predicted', methods=['GET'])
def download_predicted():
    if 'username' not in session:
        return "Unauthorized", 401
    username = session['username']

    user_pred_folder = os.path.join(PRED_FOLDER, username)
    if not os.path.exists(user_pred_folder):
        return "Predicted folder not found", 404

    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, 'w') as zipf:
        for filename in os.listdir(user_pred_folder):
            file_path = os.path.join(user_pred_folder, filename)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=filename)

    zip_stream.seek(0)
    return send_file(zip_stream, mimetype='application/zip', as_attachment=True, download_name='Predicted_Labels.zip')


# @app.route('/download_human', methods=['POST'])
# def download_human():
#     data = request.get_json()
#     file_paths = data.get('files', [])

#     if not file_paths:
#         return "No files provided", 400

#     zip_stream = io.BytesIO()
#     with zipfile.ZipFile(zip_stream, 'w') as zipf:
#         for path in file_paths:
#             if os.path.exists(path):
#                 arcname = os.path.basename(path)
#                 zipf.write(path, arcname=arcname)

#     zip_stream.seek(0)
#     return send_file(zip_stream, mimetype='application/zip', as_attachment=True, download_name='Human_Labels.zip')

@app.route('/download_human', methods=['POST'])
def download_human():
    if 'username' not in session:
        return "Unauthorized", 401
    username = session['username']
    user_human_folder = os.path.join(HUMAN_LABEL_FOLDER, username)

    data = request.get_json()
    file_paths = data.get('files', [])

    if not file_paths:
        return "No files provided", 400

    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, 'w') as zipf:
        for full_path in file_paths:
            if os.path.exists(full_path) and full_path.startswith(user_human_folder):
                arcname = os.path.basename(full_path)
                zipf.write(full_path, arcname=arcname)

    zip_stream.seek(0)
    return send_file(zip_stream, mimetype='application/zip', as_attachment=True, download_name='Human_Labels.zip')

@app.route('/login_page')
def login_page():
    return render_template('login.html')

# @app.route('/copy_predictions_to_human', methods=['POST'])
# def copy_predictions_to_human():
#     os.makedirs(PRED_FOLDER, exist_ok=True)
#     os.makedirs(HUMAN_LABEL_FOLDER, exist_ok=True)

#     copied_files = []

#     for fname in os.listdir(PRED_FOLDER):
#         if fname.endswith('.mat'):
#             src = os.path.join(PRED_FOLDER, fname)
#             dst = os.path.join(HUMAN_LABEL_FOLDER, fname)

#             if not os.path.exists(dst):
#                 shutil.copy(src, dst)
#                 copied_files.append(fname)

#     return jsonify({"status": "done", "copied": copied_files})



@app.route('/copy_predictions_to_human', methods=['POST'])
def copy_predictions_to_human():
    if 'username' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    username = session['username']

    user_pred_folder = os.path.join(PRED_FOLDER, username)
    user_human_folder = os.path.join(HUMAN_LABEL_FOLDER, username)

    os.makedirs(user_pred_folder, exist_ok=True)
    os.makedirs(user_human_folder, exist_ok=True)

    copied_files = []

    for fname in os.listdir(user_pred_folder):
        if fname.endswith('.mat'):
            src = os.path.join(user_pred_folder, fname)
            dst = os.path.join(user_human_folder, fname)

            if not os.path.exists(dst):
                shutil.copy(src, dst)
                copied_files.append(fname)

    return jsonify({"status": "done", "copied": copied_files})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['username'] = username
        return jsonify({'success': True})
    else:
        return jsonify({'success': False})



@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()

    return jsonify({'success': success})


if __name__ == '__main__':
    from pyngrok import ngrok
    public_url = ngrok.connect(5000)
    print(" * ngrok tunnel:", public_url)
    app.run(debug=False)
