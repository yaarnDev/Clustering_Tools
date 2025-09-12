from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import zipfile
import io
import glob

app = Flask(__name__)

def manual_kmeans(data, n_clusters=3, max_iter=300, random_state=42, n_init=10):
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


@app.route('/preview_excel', methods=['POST'])
def preview_excel():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = file.filename
    expected_columns = ["No", "Nama", "NIM", "Alpro", "Struktur Data", "Basis Data", "Dasar P.Web"]

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')), nrows=5)
        else:
            df = pd.read_excel(file, nrows=5)
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            return f"""<div class="bg-red-800/20 text-red-300 p-4 rounded-lg border border-red-800/50">
                        <p class="font-semibold mb-2">Kolom tidak lengkap!</p>
                        <p class="text-sm">File Anda harus memiliki kolom berikut: {', '.join(missing_cols)}.</p>
                      </div>""", 400
        
        return df.to_html(classes="preview-table", index=False)

    except Exception as e:
        return f"""<div class="bg-red-800/20 text-red-300 p-4 rounded-lg border border-red-800/50">
                    <p class="font-semibold mb-2">Terjadi kesalahan saat mempratinjau file.</p>
                    <p class="text-sm">Detail: {str(e)}</p>
                  </div>""", 500


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    try:
        plot_dir = 'static/plots'
        if os.path.exists(plot_dir):
            files = glob.glob(os.path.join(plot_dir, '*'))
            for f in files:
                os.remove(f)
    except Exception as e:
        print(f"Error saat membersihkan folder plots: {e}")
    
    try:
        if os.path.exists('hasil_clustering.csv'):
            os.remove('hasil_clustering.csv')
    except Exception as e:
        print(f"Error saat membersihkan file hasil_clustering.csv: {e}")

    file = request.files['file']
    if not file:
        return "No file uploaded", 400

    filename = file.filename
    if filename.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

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
    plt.figure()
    sns.scatterplot(x=data_normalized[:, 0], y=data_normalized[:, 1], hue=df['Kategori'], palette='viridis')
    plt.title("Visualisasi Clustering Mahasiswa")
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.savefig(scatter_path)
    plt.close()

    hist_paths = []
    for i, feature in enumerate(selected_features):
        path = f"static/plots/hist_{i}_{plot_id}.png"
        plt.figure()
        plt.hist(df[feature], bins=10, edgecolor='black')
        plt.title(f"Distribusi {feature}")
        plt.savefig(path)
        hist_paths.append(path)
        plt.close()

    boxplot_path = f"static/plots/boxplot_{plot_id}.png"
    plt.figure()
    sns.boxplot(x=df['Kategori'], y=df[selected_features[0]], palette='coolwarm')
    plt.title(f"Boxplot {selected_features[0]} berdasarkan Kategori Clustering")
    plt.savefig(boxplot_path)
    plt.close()

    pie_path = f"static/plots/pie_{plot_id}.png"
    plt.figure()
    df['Kategori'].value_counts().plot.pie(autopct='%1.1f%%', colors=['red', 'yellow', 'green'])
    plt.title("Proporsi Mahasiswa dalam Setiap Cluster")
    plt.ylabel("")
    plt.savefig(pie_path)
    plt.close()
    
    heatmap_path = f"static/plots/heatmap_{plot_id}.png"
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_normalized[selected_features], annot=True, cmap='YlGnBu', cbar=True)
    plt.title("Ilustrasi Normalisasi Nilai Praktikum Mahasiswa (Min-Max Normalization)")
    plt.xlabel("Mata Kuliah")
    plt.ylabel("Mahasiswa")
    plt.savefig(heatmap_path)
    plt.close()

    hasil_path = 'hasil_clustering.csv'
    df_sorted.to_csv(hasil_path, index=False)

    return render_template("result.html",
                           tables=[df_sorted.to_html(classes="table table-bordered", index=False)],
                           normalized=df_normalized.to_html(classes="table table-sm table-striped", index=False),
                           score=silhouette,
                           inertia_score=inertia,
                           scatter_plot=scatter_path,
                           histograms=hist_paths,
                           boxplot=boxplot_path,
                           pie_chart=pie_path,
                           heatmap_plot=heatmap_path)


@app.route('/download')
def download():
    file_path = os.path.join(os.getcwd(), 'hasil_clustering.csv')
    if not os.path.exists(file_path):
        return "File hasil clustering tidak ditemukan.", 404
        
    return send_file(file_path, as_attachment=True, download_name='hasil_clustering.csv')


@app.route('/download_histograms')
def download_histograms():
    hist_files = [f for f in os.listdir('static/plots') if f.startswith('hist_')]
    
    if not hist_files:
        return "Tidak ada file histogram yang dapat diunduh.", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in hist_files:
            file_path = os.path.join('static/plots', filename)
            zf.write(file_path, filename)
            os.remove(file_path)
            
    memory_file.seek(0)

    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='histograms.zip')


if __name__ == '__main__':
    app.run(debug=True)