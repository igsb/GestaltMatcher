## gestaltmatcher_server.py
# HTTP server wrapping GestaltMatcher encode and analysis pipelines.
# Mirrors the cropper_server.py pattern.

# This is intended to run in a Docker container, and is not
# optimized/secured for production use outside of that context.
    
import io
import os
import json
import base64
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingMixIn, TCPServer

import matplotlib
matplotlib.use('Agg')  # headless backend — must come before any pyplot import

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from lib.models.deep_gestalt import DeepGestalt
from lib.datasets.utils import load_synds_list, load_deep_gestalt_encodings
from lib.evaluation.distance import calculate_distance
from lib.evaluation.visualization import plot_tsne, plot_clustering_heatmap
import matplotlib.pyplot as plt

# --- Configuration --------------------------------------------------------------

GM_PORT          = int(os.environ.get('GM_PORT') or '5000')
GM_HOST          = os.environ.get('GM_HOST') or '0.0.0.0'
GM_MODEL         = (os.environ.get('GM_MODEL') or
                    './saved_models/s101_gmdb_aug_adam_DeepGestalt_e355_ReLU_bs280.pt')
GM_CONTROL_EMBS  = os.environ.get('GM_CONTROL_EMBS') or './tables/gmdb_encodings_v1.0.1.csv'
GM_LOOKUP_TABLE  = os.environ.get('GM_LOOKUP_TABLE') or './tables/lookup_table.txt'
# --- Model startup (runs once at import time) -----------------------------------

print(f"Loading model from {GM_MODEL} ...", flush=True)
device     = torch.device('cpu')
state_dict = torch.load(GM_MODEL, map_location=device)
num_classes = state_dict['classifier.0.weight'].shape[0]
model = DeepGestalt(
    in_channels=1,
    num_classes=num_classes,
    device=device,
    pretrained=False,
    act_type=nn.ReLU,
).to(device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.", flush=True)

synd_list  = load_synds_list(GM_LOOKUP_TABLE)
gmdb_data  = load_deep_gestalt_encodings(GM_CONTROL_EMBS, synd_list)
print("Control embeddings loaded.", flush=True)

# --- Inference helpers -----------------------------------------------------------

# Verbatim copy of preprocess() from predict.py lines 23-29
def preprocess(img):
    resize = transforms.Resize((100, 100))  # Default size is (100,100)
    img = resize(img)

    # desired number of channels is 1, so we convert to gray
    img = transforms.Grayscale(1)(img)
    return transforms.ToTensor()(img)


def encode_image(image_bytes, filename):
    """Run DeepGestalt inference on raw image bytes, return CSV embedding row."""
    img = Image.open(io.BytesIO(image_bytes))
    img_tensor = preprocess(img).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred, pred_rep = model(img_tensor.unsqueeze(0))
    row = f"{filename};{pred.squeeze().tolist()};{pred_rep.squeeze().tolist()}"
    return f"img_name;class_conf;representations\n{row}\n"


def run_analysis(metadata_str, embs_str, exp_name, linkage, output_image_type):
    """
    Run full GestaltMatcher analysis pipeline.
    Inputs are passed as strings; all intermediate files live in a temp dir.
    Returns a dict with base64-encoded images and TSV text.
    """
    np.random.seed(1027)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write inputs to temp files
        meta_path = os.path.join(tmpdir, 'target_metadata.tsv')
        embs_path = os.path.join(tmpdir, 'target_embs.csv')
        with open(meta_path, 'w') as f:
            f.write(metadata_str)
        with open(embs_path, 'w') as f:
            f.write(embs_str)

        # Load target embeddings
        data = load_deep_gestalt_encodings(embs_path, synd_list)
        target_df = pd.read_csv(meta_path, sep='\t')
        labels           = target_df.label.values
        class_labels     = target_df.class_label.values
        target_image_ids = target_df.image_id.values.astype(str)
        target_embeddings = np.array([data['embeddings'][i] for i in target_image_ids])

        # Control embeddings (already loaded at startup; exclude target IDs)
        gmdb_image_ids = gmdb_data['image_ids']
        gmdb_image_ids = np.array([i for i in gmdb_image_ids if i not in target_image_ids])
        gmdb_embeddings = np.array([gmdb_data['embeddings'][i] for i in gmdb_image_ids])

        # Pairwise distance matrix (target × target)
        distance = calculate_distance(target_embeddings, target_embeddings)
        df = pd.DataFrame(distance, columns=target_image_ids, index=target_image_ids)
        output_df_path = os.path.join(tmpdir, 'distance_matrix.tsv')
        df.to_csv(output_df_path, sep='\t')

        # Annotation sizing (mirrors gestalt_matcher_analysis.plot_distance_matrix)
        ann_size, tick_size, rotation = 20, 18, 0
        if len(df.columns) >= 40:
            ann_size, tick_size, rotation = 8, 8, 30
        elif len(df.columns) >= 20:
            ann_size, tick_size, rotation = 10, 10, 30
        elif len(df.columns) >= 15:
            ann_size, tick_size = 16, 14
        map_dict = {str(i): str(j) for i, j in zip(labels, df.columns.values)}

        # tSNE (target only)
        plot_tsne(target_embeddings, target_image_ids, class_labels, tmpdir,
                  syndrome_name_dict=None, show_metadata=False, synd_colors=None,
                  title=f'{exp_name}_tSNE', gallery_dot_size=260, test_dot_size=300,
                  file_type=output_image_type, marker_dict=None, perplexity=15, not_show=True)

        # Pairwise distance heatmap
        plot_clustering_heatmap(df, df, labels, target_image_ids, exp_name,
                                tmpdir, output_image_type,
                                ann_size, tick_size, rotation, threshold=1.65, match_rank=30,
                                display_match_box=False, source_type='distance', map_dict=map_dict,
                                input_crops_path=None, linkage_method=linkage,
                                row_cluster=True, col_cluster=True,
                                file_suffix=f'_{linkage}')
        plt.close('all')

        # tSNE with control
        rand_index = np.random.randint(0, len(gmdb_embeddings), size=len(target_image_ids) * 10)
        sel_ctrl_embs = np.array(gmdb_embeddings)[rand_index]
        sel_ctrl_ids  = np.array(gmdb_image_ids)[rand_index]
        all_image_ids = np.append(target_image_ids, sel_ctrl_ids)
        all_labels    = np.array(list(class_labels) + ['Control'] * len(sel_ctrl_embs))
        plot_tsne(np.append(target_embeddings, sel_ctrl_embs, axis=0), all_image_ids,
                  all_labels, tmpdir,
                  syndrome_name_dict=None, show_metadata=False, synd_colors=None,
                  title=f'{exp_name}_tSNE_with_control', gallery_dot_size=260, test_dot_size=300,
                  file_type=output_image_type, marker_dict=None, perplexity=15, not_show=True)

        # Pairwise rank matrix (target vs. full cohort)
        cohort_image_ids  = np.append(target_image_ids, gmdb_image_ids)
        cohort_embeddings = np.append(target_embeddings, gmdb_embeddings, axis=0)
        all_distances     = calculate_distance(target_embeddings, cohort_embeddings)
        target_ranks = []
        for index, _image_id in enumerate(target_image_ids):
            distances             = all_distances[index]
            sorted_distance_indices = np.argsort(distances)
            sorted_image_ids      = cohort_image_ids[sorted_distance_indices]
            ranks = [int(np.where(sorted_image_ids == tid)[0][0]) for tid in target_image_ids]
            target_ranks.append(ranks)
        target_ranks    = np.array(target_ranks)
        target_ranks_df = pd.DataFrame(target_ranks,
                                       columns=target_image_ids,
                                       index=target_image_ids).T
        output_rank_df_path = os.path.join(tmpdir, 'rank_matrix.tsv')
        target_ranks_df.to_csv(output_rank_df_path, sep='\t')

        # Pairwise rank heatmap
        plot_clustering_heatmap(df, target_ranks_df, labels, target_image_ids, exp_name,
                                tmpdir, output_image_type,
                                ann_size, tick_size, rotation, threshold=1.65, match_rank=30,
                                display_match_box=False, source_type='rank', map_dict=map_dict,
                                input_crops_path=None, linkage_method=linkage,
                                row_cluster=True, col_cluster=True,
                                file_suffix=f'_{linkage}')
        plt.close('all')

        # Read outputs and return
        def _b64(path):
            with open(path, 'rb') as fh:
                return base64.b64encode(fh.read()).decode('utf-8')

        def _text(path):
            with open(path, 'r') as fh:
                return fh.read()

        return {
            'tsne_png':             _b64(os.path.join(tmpdir, f'{exp_name}_tSNE.{output_image_type}')),
            'tsne_control_png':     _b64(os.path.join(tmpdir, f'{exp_name}_tSNE_with_control.{output_image_type}')),
            'heatmap_distance_png': _b64(os.path.join(tmpdir, f'{exp_name}_validation_pairwise_distance_{linkage}.{output_image_type}')),
            'heatmap_rank_png':     _b64(os.path.join(tmpdir, f'{exp_name}_validation_pairwise_rank_{linkage}.{output_image_type}')),
            'distance_matrix':      _text(output_df_path),
            'rank_matrix':          _text(output_rank_df_path),
        }


# --- HTTP handler --------------------------------------------------------------

class GestaltMatcherHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

    def _send_json(self, code, body):
        data = json.dumps(body).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == '/health':
            self._send_json(200, {'status': 'ok'})
        else:
            self._send_json(404, {'status': 'error', 'message': 'not found'})

    def do_POST(self):
        try:
            body = self._read_json()

            if self.path == '/encode':
                image_bytes = base64.b64decode(body['image_base64'])
                filename    = body['filename']
                embedding   = encode_image(image_bytes, filename)
                self._send_json(200, {'status': 'ok', 'embedding': embedding})

            elif self.path == '/analyse':
                result = run_analysis(
                    metadata_str      = body['target_metadata'],
                    embs_str          = body['target_embs'],
                    exp_name          = body['exp_name'],
                    linkage           = body.get('linkage', 'average'),
                    output_image_type = body.get('output_image_type', 'png'),
                )
                self._send_json(200, {'status': 'ok', **result})

            else:
                self._send_json(404, {'status': 'error', 'message': 'not found'})

        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {'status': 'error', 'message': str(e)})


class ThreadedHTTPServer(ThreadingMixIn, TCPServer):
    allow_reuse_address = True
    daemon_threads      = True


if __name__ == '__main__':
    server = ThreadedHTTPServer((GM_HOST, GM_PORT), GestaltMatcherHandler)
    print(f"Model loaded. Listening on {GM_HOST}:{GM_PORT}", flush=True)
    server.serve_forever()
