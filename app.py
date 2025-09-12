# =================================================================
# BAGIAN 1: IMPORT DAN SETUP APLIKASI
# =================================================================
import os
import io
import uuid
import glob
import zipfile
import datetime
import bcrypt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Mode non-interaktif untuk Matplotlib di server
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

# Setup Awal
load_dotenv()
app = Flask(__name__)
# Pastikan Anda sudah membuat file .env dengan SECRET_KEY di dalamnya
app.secret_key = os.getenv("SECRET_KEY")

# Koneksi ke MongoDB Atlas
# Pastikan Anda sudah membuat file .env dengan MONGO_URI di dalamnya
client = MongoClient(os.getenv("MONGO_URI"))
db = client.db_clustering 
users_collection = db.users
history_collection = db.history

# =================================================================
# BAGIAN 2: FUNGSI-FUNGSI BANTUAN (LOGIKA CLUSTERING ANDA)
# =================================================================

def manual_kmeans(data, n_clusters=3, max_iter=300, random_state=42, n_init=10):
    # ... (Kode fungsi manual_kmeans Anda tidak berubah) ...
    best_clusters = None
    best_inertia = float('inf')
    for _ in range(n_init):
        np.random.seed(random_state + _)
        centroids_indices = np.random.choice(data.shape[0], n_clusters, replace=False)
        centroids = data[centroids_indices]
        for i in range(max_iter):
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            clusters = np.argmin(distances, axis=0)
            new_centroids = np.array([data[clusters == j].mean(axis=0) for j in range(n_clusters)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        inertia = np.sum([np.sum((data[clusters == j] - centroids[j])**2) for j in range(n_clusters)])
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters
    return best_clusters, best_inertia

# =================================================================
# BAGIAN 3: ROUTE UNTUK OTENTIKASI (LOGIN, REGISTER, LOGOUT)
# =================================================================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # ... (Kode login tidak berubah) ...
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
    # ... (Kode register tidak berubah) ...
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

# =================================================================
# BAGIAN 4: ROUTE UTAMA APLIKASI (DASHBOARD & CLUSTERING)
# =================================================================

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
    
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = file.filename
    file_content = io.BytesIO(file.read())
    
    # ... (Sisa kode proses clustering Anda SAMA seperti sebelumnya, tidak ada yang diubah) ...
    if filename.endswith('.csv'):
        df = pd.read_csv(file_content)
    else:
        df = pd.read_excel(file_content)
    
    expected_columns = ["No", "Nama", "NIM", "Alpro", "Struktur Data", "Basis Data", "Dasar P.Web"]
    missing_cols = [col for col in expected_columns if col not in df.columns]

    if missing_cols:
        return f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}", 400

    selected_features = ["Alpro", "Struktur Data", "Basis Data", "Dasar P.Web"]
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(df[selected_features])
    df_normalized = pd.DataFrame(data_normalized, columns=selected_features)
    df_normalized.insert(0, "Nama", df["Nama"])
    df_normalized.insert(0, "NIM", df["NIM"])
    df_normalized.insert(0, "No", df["No"])

    clusters, inertia = manual_kmeans(data_normalized, n_clusters=3, random_state=42, n_init=10)
    cluster_means = pd.DataFrame(data_normalized, columns=selected_features)
    cluster_means['Cluster'] = clusters
    mean_per_cluster = cluster_means.groupby('Cluster').mean().mean(axis=1)
    ordered_clusters = mean_per_cluster.sort_values(ascending=False).index.tolist()
    cluster_mapping = {ordered_clusters[0]: 2, ordered_clusters[1]: 1, ordered_clusters[2]: 0}
    df['Cluster'] = [cluster_mapping[c] for c in clusters]
    kategori_map = {2: 'Tinggi', 1: 'Sedang', 0: 'Rendah'}
    df['Kategori'] = df['Cluster'].map(kategori_map)
    
    df_sorted = df.sort_values('Cluster', ascending=False)
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['No'] = df_sorted.index + 1

    silhouette = round(silhouette_score(data_normalized, clusters), 4)
    inertia = round(inertia, 4)

    os.makedirs("static/plots", exist_ok=True)
    plot_id = str(uuid.uuid4())
    
    scatter_path = f"static/plots/scatter_{plot_id}.png"
    plt.figure(); sns.scatterplot(x=data_normalized[:, 0], y=data_normalized[:, 1], hue=df['Kategori'], palette='viridis'); plt.title("Visualisasi Clustering Mahasiswa"); plt.xlabel(selected_features[0]); plt.ylabel(selected_features[1]); plt.savefig(scatter_path); plt.close()

    hist_paths = []
    for i, feature in enumerate(selected_features):
        path = f"static/plots/hist_{i}_{plot_id}.png"; plt.figure(); plt.hist(df[feature], bins=10, edgecolor='black'); plt.title(f"Distribusi {feature}"); plt.savefig(path); hist_paths.append(path); plt.close()

    boxplot_path = f"static/plots/boxplot_{plot_id}.png"; plt.figure(); sns.boxplot(x=df['Kategori'], y=df[selected_features[0]], palette='coolwarm'); plt.title(f"Boxplot {selected_features[0]} berdasarkan Kategori Clustering"); plt.savefig(boxplot_path); plt.close()
    
    pie_path = f"static/plots/pie_{plot_id}.png"; plt.figure(); df['Kategori'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'yellow', 'green']); plt.title("Proporsi Mahasiswa dalam Setiap Cluster"); plt.ylabel(""); plt.savefig(pie_path); plt.close()
    
    heatmap_path = f"static/plots/heatmap_{plot_id}.png"; plt.figure(figsize=(10, 6)); sns.heatmap(df_normalized[selected_features], annot=True, cmap='YlGnBu', cbar=True); plt.title("Ilustrasi Normalisasi Nilai Praktikum Mahasiswa"); plt.xlabel("Mata Kuliah"); plt.ylabel("Mahasiswa"); plt.savefig(heatmap_path); plt.close()

    hasil_path = 'hasil_clustering.csv'; df_sorted.to_csv(hasil_path, index=False)

    hasil_json = df_sorted.to_dict(orient='records')
    history_record = {
        "user_id": ObjectId(session['user_id']), "username": session['username'], "filename": filename,
        "cluster_results_summary": {"silhouette_score": silhouette, "inertia": inertia, "total_data": len(df_sorted)},
        "results_data": hasil_json,
        "plot_paths": {"scatter": scatter_path, "histograms": hist_paths, "boxplot": boxplot_path, "pie": pie_path, "heatmap": heatmap_path},
        "timestamp": datetime.datetime.utcnow()
    }
    history_collection.insert_one(history_record)

    return render_template("result.html", tables=[df_sorted.to_html(classes="table table-bordered", index=False)], normalized=df_normalized.to_html(classes="table table-sm table-striped", index=False), score=silhouette, inertia_score=inertia, scatter_plot=scatter_path, histograms=hist_paths, boxplot=boxplot_path, pie_chart=pie_path, heatmap_plot=heatmap_path)


@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_history_cursor = history_collection.find(
        {'user_id': ObjectId(session['user_id'])}
    ).sort('timestamp', -1)
    
    # [PERUBAHAN] Mengubah cursor menjadi list dan menambahkan 'id'
    # Ini agar template bisa mengakses ID unik (_id) untuk tombol hapus
    history_list = []
    for record in user_history_cursor:
        record['id'] = str(record['_id']) # Mengonversi ObjectId ke string
        history_list.append(record)
    
    return render_template('history.html', history_data=history_list)


# =================================================================
# BAGIAN 5: [BARU] ROUTE UNTUK MENGHAPUS HISTORY
# =================================================================
@app.route('/delete_history/<history_id>', methods=['POST'])
def delete_history(history_id):
    if 'user_id' not in session:
        flash('Anda harus login untuk melakukan aksi ini.', 'danger')
        return redirect(url_for('login'))

    try:
        # 1. Cari record di MongoDB berdasarkan ID dalam bentuk string
        record_to_delete = history_collection.find_one({'_id': ObjectId(history_id)})
        
        if not record_to_delete:
            flash('Riwayat tidak ditemukan.', 'danger')
            return redirect(url_for('history'))

        # 2. Pastikan record ini milik user yang sedang login (keamanan)
        if record_to_delete['user_id'] != ObjectId(session['user_id']):
            flash('Anda tidak memiliki izin untuk menghapus riwayat ini.', 'danger')
            return redirect(url_for('history'))

        # 3. Hapus file-file plot yang terkait dari folder static
        if 'plot_paths' in record_to_delete and record_to_delete['plot_paths']:
            plot_paths = record_to_delete['plot_paths']
            
            # Loop melalui semua path di dictionary plot_paths
            for key, path_or_list in plot_paths.items():
                if isinstance(path_or_list, list): # Kasus khusus untuk histograms
                    for path in path_or_list:
                        full_path = os.path.join(app.root_path, path)
                        if os.path.exists(full_path):
                            os.remove(full_path)
                else: # Untuk path tunggal seperti scatter, pie, dll.
                    full_path = os.path.join(app.root_path, path_or_list)
                    if os.path.exists(full_path):
                        os.remove(full_path)
        
        # 4. Hapus record dari database MongoDB
        history_collection.delete_one({'_id': ObjectId(history_id)})

        # 5. Beri notifikasi ke user dan redirect
        flash('Riwayat berhasil dihapus.', 'success')

    except Exception as e:
        flash(f'Terjadi kesalahan saat menghapus riwayat: {e}', 'danger')

    return redirect(url_for('history'))


# =================================================================
# BAGIAN 6: ROUTE LAINNYA (Preview, Download - TIDAK BERUBAH)
# =================================================================
@app.route('/preview_excel', methods=['POST'])
def preview_excel():
    # ... (Kode /preview_excel Anda SAMA seperti sebelumnya) ...
    file = request.files['file'];
    if not file: return "No file uploaded", 400
    filename = file.filename
    expected_columns = ["No", "Nama", "NIM", "Alpro", "Struktur Data", "Basis Data", "Dasar P.Web"]
    try:
        if filename.endswith('.csv'): df = pd.read_csv(io.StringIO(file.read().decode('utf-8')), nrows=5)
        else: df = pd.read_excel(file, nrows=5)
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols: return f"""<div class="bg-red-800/20 text-red-300 p-4 rounded-lg border border-red-800/50">Kolom berikut tidak ditemukan: {', '.join(missing_cols)}</div>""", 400
        return df.to_html(classes="preview-table", index=False)
    except Exception as e:
        return f"""<div class="bg-red-800/20 text-red-300 p-4 rounded-lg border border-red-800/50">Error membaca file: {e}</div>""", 500


@app.route('/download')
def download():
    # ... (Kode /download Anda SAMA seperti sebelumnya) ...
    file_path = os.path.join(os.getcwd(), 'hasil_clustering.csv')
    if not os.path.exists(file_path):
        return "File hasil clustering tidak ditemukan.", 404
    return send_file(file_path, as_attachment=True, download_name='hasil_clustering.csv')


@app.route('/download_histograms')
def download_histograms():
    # ... (Kode /download_histograms Anda SAMA seperti sebelumnya) ...
    # Di sini perlu logika untuk mengambil path histogram dari record history terakhir jika ingin spesifik
    # Untuk sementara, ini akan mengunduh semua file hist yang ada
    hist_files = [f for f in os.listdir('static/plots') if f.startswith('hist_')]
    if not hist_files:
        return "Tidak ada file histogram yang dapat diunduh.", 404
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in hist_files:
            file_path = os.path.join('static/plots', filename)
            zf.write(file_path, filename)
    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='histograms.zip')

# =================================================================
# BAGIAN 7: MENJALANKAN APLIKASI
# =================================================================
if __name__ == '__main__':
    app.run(debug=True)