import os
import io
import uuid
import zipfile
import datetime
import bcrypt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode non-interaktif untuk Matplotlib di server
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Setup Awal
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# Koneksi ke MongoDB Atlas
client = MongoClient(os.getenv("MONGO_URI"))
db = client.db_clustering
users_collection = db.users
history_collection = db.history

def manual_kmeans(data, n_clusters=3, max_iter=300, random_state=42, n_init=10):
    best_clusters = None
    best_inertia = float('inf')
    for _ in range(n_init):
        np.random.seed(random_state + _)
        if data.shape[0] < n_clusters:
            return None, float('inf')
        centroids_indices = np.random.choice(data.shape[0], n_clusters, replace=False)
        centroids = data[centroids_indices]
        for i in range(max_iter):
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            clusters = np.argmin(distances, axis=0)
            new_centroids = np.array([data[clusters == j].mean(axis=0) for j in range(n_clusters)])
            for j in range(n_clusters):
                if np.isnan(new_centroids[j]).any():
                    new_centroids[j] = data[np.random.choice(data.shape[0])]
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        inertia = np.sum([np.sum((data[clusters == j] - centroids[j])**2) for j in range(n_clusters)])
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters
    return best_clusters, best_inertia

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users_collection.find_one({'username': username}):
            flash('Username sudah digunakan!', 'danger')
            return redirect(url_for('register'))
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        users_collection.insert_one({'username': username, 'password': hashed_password})
        flash('Registrasi berhasil! Silakan login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah berhasil logout.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    if 'user_id' not in session:
        flash('Silakan login untuk melakukan clustering.', 'danger')
        return redirect(url_for('login'))
    
    file = request.files.get('file')
    selected_features = request.form.getlist('selected_features')

    if not file:
        flash('File tidak ditemukan.', 'danger')
        return redirect(url_for('dashboard'))

    if not selected_features or len(selected_features) < 2:
        flash('Silakan pilih minimal 2 kolom untuk clustering.', 'danger')
        return redirect(url_for('dashboard'))

    filename = file.filename
    try:
        file_content = io.BytesIO(file.read())
        if filename.endswith('.csv'):
            df_original = pd.read_csv(file_content)
        else:
            # [PERBAIKAN TAMBAHAN] Menambahkan header=0 agar pembacaan file Excel lebih andal
            df_original = pd.read_excel(file_content, header=0)
    except Exception as e:
        flash(f"Gagal membaca file: {e}", "danger")
        return redirect(url_for('dashboard'))

    missing_cols = [col for col in selected_features if col not in df_original.columns]
    if missing_cols:
        flash(f"Kolom berikut tidak ditemukan di file Anda: {', '.join(missing_cols)}", 'danger')
        return redirect(url_for('dashboard'))
    
    df = df_original.copy()
    df.dropna(subset=selected_features, inplace=True)

    if df.empty:
        flash("Tidak ada data valid yang tersisa setelah menghapus baris dengan nilai kosong.", "danger")
        return redirect(url_for('dashboard'))

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(df[selected_features])
    
    clusters, inertia = manual_kmeans(data_normalized, n_clusters=3, random_state=42, n_init=10)
    
    if clusters is None:
        flash("Clustering gagal. Jumlah data valid lebih sedikit dari jumlah cluster yang diinginkan.", "danger")
        return redirect(url_for('dashboard'))
        
    df['Cluster'] = clusters
    
    cluster_means = pd.DataFrame(data_normalized, columns=selected_features)
    cluster_means['Cluster'] = clusters
    mean_per_cluster = cluster_means.groupby('Cluster').mean().mean(axis=1)
    ordered_clusters = mean_per_cluster.sort_values(ascending=False).index.tolist()
    
    cluster_mapping = {ordered_clusters[i]: 2 - i for i in range(len(ordered_clusters))}
    if len(ordered_clusters) < 3:
        if 0 not in cluster_mapping: cluster_mapping[0] = 0
        if 1 not in cluster_mapping: cluster_mapping[1] = 1
        if 2 not in cluster_mapping: cluster_mapping[2] = 2

    df['Cluster'] = df['Cluster'].map(cluster_mapping)
    kategori_map = {2: 'Tinggi', 1: 'Sedang', 0: 'Rendah'}
    df['Kategori'] = df['Cluster'].map(kategori_map)
    
    df_sorted = df.sort_values('Cluster', ascending=False).reset_index(drop=True)

    df_sorted['No'] = range(1, 1 + len(df_sorted))

    silhouette = round(silhouette_score(data_normalized, clusters), 4)
    inertia = round(inertia, 4)

    df_normalized_display = pd.DataFrame(data_normalized, columns=selected_features)
    for col_id in ['NIM', 'Nama']:
        if col_id in df.columns:
            df_normalized_display.insert(0, col_id, df.reset_index(drop=True).loc[df_normalized_display.index, col_id])
    
    df_normalized_display.insert(0, 'No', range(1, 1 + len(df_normalized_display)))

    os.makedirs("static/plots", exist_ok=True)
    plot_id = str(uuid.uuid4())
    
    scatter_path = f"static/plots/scatter_{plot_id}.png"
    plt.figure()
    sns.scatterplot(x=data_normalized[:, 0], y=data_normalized[:, 1], hue=df['Kategori'], palette='viridis')
    plt.title("Visualisasi Clustering")
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.savefig(scatter_path)
    plt.close()
    
    hist_paths = []
    for feature in selected_features:
        path = f"static/plots/hist_{feature.replace(' ', '_')}_{plot_id}.png"
        plt.figure()
        plt.hist(df[feature], bins=10, edgecolor='black')
        plt.title(f"Distribusi {feature}")
        plt.savefig(path)
        plt.close()
        hist_paths.append(path)

    boxplot_path = f"static/plots/boxplot_{plot_id}.png"
    plt.figure()
    sns.boxplot(x=df['Kategori'], y=df[selected_features[0]], palette='coolwarm')
    plt.title(f"Boxplot {selected_features[0]}")
    plt.savefig(boxplot_path)
    plt.close()
    
    pie_path = f"static/plots/pie_{plot_id}.png"
    plt.figure()
    df['Kategori'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#2ca02c', '#ff7f0e', '#d62728'])
    plt.title("Proporsi Cluster")
    plt.ylabel("")
    plt.savefig(pie_path)
    plt.close()
    
    heatmap_path = f"static/plots/heatmap_{plot_id}.png"
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df_normalized_display[selected_features], 
        annot=True, 
        fmt=".2f",
        cmap='YlGnBu',
        cbar_kws={'label': 'Nilai Normalisasi'}
    )
    plt.title("Heatmap Normalisasi Data", fontsize=16)
    plt.xlabel("Fitur", fontsize=12)
    plt.ylabel("Data", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    hasil_json = df_sorted.to_dict(orient='records')
    history_record = {
        "user_id": ObjectId(session['user_id']),
        "username": session['username'],
        "filename": filename,
        "selected_features": selected_features,
        # [PERBAIKAN UTAMA] Konversi tipe data numpy.float64 ke float standar Python
        "cluster_results_summary": {"silhouette_score": float(silhouette), "inertia": float(inertia), "total_data": len(df_sorted)},
        "results_data": hasil_json,
        "plot_paths": {"scatter": scatter_path, "histograms": hist_paths, "boxplot": boxplot_path, "pie": pie_path, "heatmap": heatmap_path},
        "timestamp": datetime.datetime.utcnow()
    }
    history_collection.insert_one(history_record)

    identifier_cols = [col for col in ['No', 'NIM', 'Nama'] if col in df_sorted.columns]
    final_display_columns = identifier_cols + selected_features + ['Cluster', 'Kategori']
    df_display = df_sorted[final_display_columns]

    df_display.to_csv('hasil_clustering.csv', index=False)

    return render_template("result.html", 
        tables=[df_display.to_html(classes="table table-bordered", index=False)], 
        normalized=df_normalized_display.to_html(classes="table table-sm table-striped", index=False), 
        score=silhouette, 
        inertia_score=inertia, 
        scatter_plot=scatter_path, 
        histograms=hist_paths, 
        boxplot=boxplot_path, 
        pie_chart=pie_path, 
        heatmap_plot=heatmap_path
    )

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_history_cursor = history_collection.find(
        {'user_id': ObjectId(session['user_id'])}
    ).sort('timestamp', -1)
    
    history_list = []
    for record in user_history_cursor:
        record['id'] = str(record['_id'])
        history_list.append(record)
    
    return render_template('history.html', history_data=history_list)

@app.route('/delete_history/<history_id>', methods=['POST'])
def delete_history(history_id):
    if 'user_id' not in session:
        flash('Anda harus login untuk melakukan aksi ini.', 'danger')
        return redirect(url_for('login'))
    try:
        record_to_delete = history_collection.find_one({'_id': ObjectId(history_id), 'user_id': ObjectId(session['user_id'])})
        if not record_to_delete:
            flash('Riwayat tidak ditemukan atau Anda tidak memiliki izin.', 'danger')
            return redirect(url_for('history'))

        if 'plot_paths' in record_to_delete and record_to_delete['plot_paths']:
            plot_paths = record_to_delete['plot_paths']
            for key, path_or_list in plot_paths.items():
                paths_to_delete = path_or_list if isinstance(path_or_list, list) else [path_or_list]
                for path in paths_to_delete:
                    full_path = os.path.join(app.root_path, path)
                    if os.path.exists(full_path):
                        os.remove(full_path)
        
        history_collection.delete_one({'_id': ObjectId(history_id)})
        flash('Riwayat berhasil dihapus.', 'success')
    except Exception as e:
        flash(f'Terjadi kesalahan saat menghapus riwayat: {e}', 'danger')
    return redirect(url_for('history'))

@app.route('/get_columns', methods=['POST'])
def get_columns():
    if 'user_id' not in session:
        return jsonify({"error": "Sesi tidak valid, silakan login ulang."}), 401
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "Tidak ada file yang diunggah."}), 400
    try:
        file.seek(0) 
        if file.filename.endswith('.csv'):
            df_header = pd.read_csv(file, nrows=0)
        else:
            df_header = pd.read_excel(file, nrows=0)
        return jsonify({"columns": df_header.columns.tolist()})
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 500

@app.route('/preview_excel', methods=['POST'])
def preview_excel():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400
    try:
        file.seek(0)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')), nrows=5)
        else:
            df = pd.read_excel(file, nrows=5)
        return df.to_html(classes="preview-table", index=False)
    except Exception as e:
        return f"""<div class="bg-red-800/20 text-red-300 p-4 rounded-lg border border-red-800/50">Error: {e}</div>""", 500

@app.route('/download')
def download():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    file_path = 'hasil_clustering.csv'
    if not os.path.exists(file_path):
        flash('File hasil tidak ditemukan. Lakukan proses clustering terlebih dahulu.', 'danger')
        return redirect(url_for('dashboard'))
        
    return send_file(file_path, as_attachment=True, download_name='Hasil_Clustering.csv')

@app.route('/download_histograms')
def download_histograms():
    hist_files = [f for f in os.listdir('static/plots') if f.startswith('hist_')]
    if not hist_files:
        flash('Tidak ada file histogram untuk diunduh.', 'warning')
        return redirect(request.referrer or url_for('dashboard'))
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in hist_files:
            file_path = os.path.join('static/plots', filename)
            zf.write(file_path, filename)
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='histograms.zip')

if __name__ == '__main__':
    app.run(debug=True)