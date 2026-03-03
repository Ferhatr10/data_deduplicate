import os
import logging
import pandas as pd
import numpy as np
import duckdb
import torch
import math
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeduplicationLayer:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', similarity_threshold=0.88, batch_size=256):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        # Cosine Similarity -> L2 distance conversion for normalized vectors:
        # L2 = sqrt(2 * (1 - cos_sim))
        # Note: The user mentioned sqrt(2 * (1 - 0.92)) but also 0.88 threshold.
        # I will use the similarity_threshold variable for consistency.
        self.l2_threshold = math.sqrt(2 * (1 - self.similarity_threshold))
        self.batch_size = batch_size
        
        # MAC MINI (M1/M2/M3) GPU DESTEĞİ
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        logger.info(f"DeduplicationLayer (Scikit-Learn) başlatıldı: {model_name} | Cihaz: {self.device}")
        logger.info(f"Eşik Değerleri: Cosine Similarity > {self.similarity_threshold} (L2 Mesafe < {self.l2_threshold:.4f})")
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings_with_cache(self, names, cache_path="data/embeddings_cache.npy"):
        """Vektörleri cache'ten okur veya GPU ile hesaplayıp kaydeder."""
        if os.path.exists(cache_path):
            logger.info(f"Cache bulundu! {cache_path} adresinden yükleniyor...")
            return np.load(cache_path)

        logger.info(f"{len(names)} kayıt için vektörler hesaplanıyor (Bu işlem 1 kez yapılır)...")
        embeddings = self.model.encode(
            names, 
            batch_size=self.batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            normalize_embeddings=True # Cosine Similarity dönüşümü için şart
        )
        
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Vektörler {cache_path} adresine kaydedildi.")
        return embeddings

    def find_matches_sklearn(self, unique_ids, embeddings, k=5, match_cache="data/matches_cache.parquet"):
        """Scikit-Learn NearestNeighbors (BallTree) kullanarak güvenli arama yapar. Cache desteği eklendi."""
        
        # 0. EĞER ÖNCEDEN HESAPLANDIYSA DİSKTEN OKU
        if os.path.exists(match_cache):
            logger.info(f"Eşleşme cache'i bulundu ({match_cache}), BallTree atlanıyor...")
            # Bu fonksiyon liste beklediği için .values.tolist() ile dönüyoruz
            return pd.read_parquet(match_cache).values.tolist()

        logger.info(f"NearestNeighbors (BallTree) kuruluyor (k={k}, metric=l2)...")
        # Veriyi float32 ve contiguous hale getir
        data = np.ascontiguousarray(embeddings).astype('float32')
        
        # BallTree büyük veri setleri için IndexFlat'ten daha stabil olabilir
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='l2', n_jobs=-1)
        nn.fit(data)
        
        logger.info("Benzerlik taranıyor...")
        distances, indices = nn.kneighbors(data)
        
        match_pairs = []
        n = len(unique_ids)
        for i in range(n):
            for j in range(1, k + 1): # İlk sonuç kendisi
                dist = distances[i][j]
                if dist < self.l2_threshold:
                    idx_r = indices[i][j]
                    if idx_r == -1 or idx_r >= n: continue
                    
                    id_l, id_r = unique_ids[i], unique_ids[idx_r]
                    if id_l < id_r:
                        match_pairs.append((id_l, id_r))
        
        # 5. BULUNANLARI KAYDET (BİR SONRAKİ HATA İÇİN ÖNLEM)
        os.makedirs(os.path.dirname(match_cache), exist_ok=True)
        pd.DataFrame(match_pairs, columns=['id_l', 'id_r']).to_parquet(match_cache)
        
        logger.info(f"Toplam {len(match_pairs)} eşleşme bulundu ve {match_cache} dosyasına kaydedildi.")
        return match_pairs

    def run_deduplication(self, input_path, output_path, metrics=None):
        logger.info(f"Loading data from {input_path}")
        con = duckdb.connect()
        # CAST(unique_id AS VARCHAR) is clean, unlike pandas .astype(str) for bytes
        query = f"SELECT * EXCLUDE(unique_id), CAST(unique_id AS VARCHAR) as unique_id FROM read_parquet('{input_path}')"
        df = con.execute(query).df()
        
        # ID ve İsim tutarlılığı
        unique_ids = df['unique_id'].tolist()
        company_names = df['company_name'].fillna('').tolist()

        # 1. Vektörleri Al (Cache + MPS/GPU)
        embeddings = self.generate_embeddings_with_cache(company_names)

        # 2. Scikit-Learn ile Arama
        match_pairs = self.find_matches_sklearn(unique_ids, embeddings)

        # 3. Clustering (Gruplama)
        logger.info("NetworkX ile kümeler (clusters) oluşturuluyor...")
        G = nx.Graph()
        G.add_nodes_from(unique_ids)
        G.add_edges_from(match_pairs)
        
        clusters = list(nx.connected_components(G))
        cluster_map = {uid: f"v_{i:06d}" for i, cluster in enumerate(clusters) for uid in cluster}
        df['cluster_id'] = df['unique_id'].map(cluster_map)

        # 4. GOLDEN RECORD AGGREGATION
        logger.info("Altın Kayıtlar (Golden Records) toparlanıyor...")
        con.register("df_clustered", df)
        golden_table_sql = """
            SELECT 
                cluster_id as canonical_id,
                MODE(company_name) as primary_company_name,
                ARRAY_AGG(DISTINCT country) FILTER (WHERE country IS NOT NULL) as operating_countries,
                LIST_FILTER(ARRAY_AGG(DISTINCT original_name), x -> x IS NOT NULL AND LOWER(TRIM(x)) <> LOWER(TRIM(MODE(company_name)))) as aliases,
                MODE(website) as primary_website,
                count(*) as record_count
            FROM df_clustered
            GROUP BY cluster_id
        """
        golden_table = con.execute(golden_table_sql).df()

        # 5. Kaydet
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        golden_out = os.path.join(os.path.dirname(os.path.dirname(output_path)), "output", "golden_table.parquet")
        os.makedirs(os.path.dirname(golden_out), exist_ok=True)
        golden_table.to_parquet(golden_out, index=False)
        
        logger.info(f"Deduplication tamamlandı! Golden Table: {golden_out}")
        
        if metrics:
            metrics.set_metric("dedupe_input_records", len(df))
            metrics.set_metric("dedupe_total_clusters", len(golden_table))
            metrics.set_metric("vector_model", self.model_name)
            
        return output_path

if __name__ == "__main__":
    layer = DeduplicationLayer(similarity_threshold=0.88)
    try:
        layer.run_deduplication("data/processed/standardized.parquet", "data/processed/clusters.parquet")
    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise e